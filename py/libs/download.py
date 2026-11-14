"""Bounded downloads for URLs supplied to the HTTP API.

Validate the actual DNS answers used by the connector, rather than performing
a separate lookup that could be bypassed by DNS rebinding. Environment proxies
are deliberately disabled so they cannot bypass this policy.
"""
import asyncio
import ipaddress
import os
import socket
import tempfile
from urllib.parse import unquote

import aiohttp
from yarl import URL


MAX_DOWNLOAD_BYTES = 50 * 1024 ** 3
MAX_REDIRECTS = 5
DOWNLOAD_TIMEOUT = aiohttp.ClientTimeout(total=1800, sock_connect=10, sock_read=30)


class UnsafeDownload(ValueError):
    pass


def _check_address(value):
    try:
        address = ipaddress.ip_address(value)
    except ValueError as exc:
        raise UnsafeDownload("Noncanonical download addresses are not allowed") from exc
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped:
        address = address.ipv4_mapped
    if not address.is_global or address.is_multicast:
        raise UnsafeDownload("Downloads must use public Internet addresses")


def validate_download_url(value):
    try:
        url = URL(value)
        if (url.scheme not in ("http", "https") or not url.host
                or url.user is not None or url.password is not None):
            raise ValueError("Invalid download URL")
        # Accessing port also validates malformed port strings.
        if url.port is None or not 1 <= url.port <= 65535:
            raise ValueError("Invalid download port")
        try:
            ipaddress.ip_address(url.host)
        except ValueError:
            if ":" in url.host or all(part.isdecimal() for part in url.host.split(".")):
                raise UnsafeDownload("Noncanonical download addresses are not allowed")
            # Hostnames are checked by PublicResolver at connection time.
        else:
            _check_address(url.host)
        return url
    except (TypeError, ValueError) as exc:
        raise UnsafeDownload("Only public HTTP(S) download URLs are allowed") from exc


class PublicResolver(aiohttp.abc.AbstractResolver):
    def __init__(self):
        self._resolver = aiohttp.resolver.DefaultResolver()

    async def resolve(self, host, port=0, family=socket.AF_INET):
        answers = await self._resolver.resolve(host, port, family)
        if not answers:
            raise UnsafeDownload("Download host has no addresses")
        for answer in answers:
            _check_address(answer["host"])
        return answers

    async def close(self):
        await self._resolver.close()


class PublicTCPConnector(aiohttp.TCPConnector):
    async def _resolve_host(self, *args, **kwargs):
        # aiohttp may recognize a literal that ipaddress rejects (e.g. an IPv4
        # address with leading zeroes) and bypass the resolver entirely. Check
        # every address returned by the connector before it can be connected.
        answers = await super()._resolve_host(*args, **kwargs)
        for answer in answers:
            _check_address(answer["host"])
        return answers


async def download_model_file(value, directory):
    url = validate_download_url(value)
    deadline = asyncio.get_running_loop().time() + DOWNLOAD_TIMEOUT.total
    try:
        return await _download_model_file(url, directory, deadline)
    except (aiohttp.ClientError, asyncio.TimeoutError):
        if url.host != "huggingface.co":
            raise
        # Preserve the existing Hugging Face mirror fallback, subject to the
        # same address policy and shared timeout budget as the original URL.
        return await _download_model_file(url.with_host("hf-mirror.com"), directory, deadline)


async def _download_model_file(value, directory, deadline):
    url = validate_download_url(value)
    filename = unquote(url.raw_path.rsplit("/", 1)[-1])
    if (not filename or filename in (".", "..")
            or any(char in filename for char in ("/", "\\", ":", "\x00"))):
        raise UnsafeDownload("Invalid download filename")

    root = os.path.realpath(directory)
    os.makedirs(root, exist_ok=True)
    destination = os.path.join(root, filename)
    if os.path.lexists(destination):
        if os.path.islink(destination) or not os.path.isfile(destination):
            raise UnsafeDownload("Invalid download destination")
        return destination

    resolver = PublicResolver()
    staged_path = None
    try:
        connector = PublicTCPConnector(resolver=resolver, use_dns_cache=False)
        async with aiohttp.ClientSession(
                connector=connector, timeout=DOWNLOAD_TIMEOUT,
                trust_env=False, auto_decompress=False,
                headers={"Accept-Encoding": "identity"}) as session:
            # Bound the entire redirect chain, not just individual requests.
            for redirects in range(MAX_REDIRECTS + 1):
                url = validate_download_url(url)
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    raise asyncio.TimeoutError()
                timeout = aiohttp.ClientTimeout(total=remaining, sock_connect=10, sock_read=30)
                async with session.get(url, allow_redirects=False, timeout=timeout) as response:
                    if response.status in (301, 302, 303, 307, 308):
                        location = response.headers.get("Location")
                        if redirects == MAX_REDIRECTS or not location:
                            raise UnsafeDownload("Invalid or excessive download redirects")
                        url = url.join(URL(location))
                        continue
                    response.raise_for_status()
                    if response.headers.get("Content-Encoding", "identity").lower() != "identity":
                        raise UnsafeDownload("Encoded model download responses are not supported")
                    if response.content_length is not None and response.content_length > MAX_DOWNLOAD_BYTES:
                        raise UnsafeDownload("Model download exceeds the size limit")
                    with tempfile.NamedTemporaryFile(
                            dir=root, prefix=".easyuse-download-", delete=False) as staged:
                        staged_path = staged.name
                        size = 0
                        async for chunk in response.content.iter_chunked(1024 * 1024):
                            size += len(chunk)
                            if size > MAX_DOWNLOAD_BYTES:
                                raise UnsafeDownload("Model download exceeds the size limit")
                            await asyncio.to_thread(staged.write, chunk)
                    os.replace(staged_path, destination)
                    return destination
    finally:
        await resolver.close()
        if staged_path and os.path.exists(staged_path):
            os.unlink(staged_path)
