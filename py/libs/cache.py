cache = {}
cache_count = {}


def update_cache(k, v):
    cache[k] = v
    cnt = cache_count.get(k)
    if cnt is None:
        cnt = 0
        cache_count[k] = cnt
    else:
        cache_count[k] += 1