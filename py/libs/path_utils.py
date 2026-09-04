import os


def resolve_output_file_path(output_root, output_file_path, file_name, file_extension):
    """Resolve a workflow-provided output path beneath ``output_root``.

    Relative output directories remain supported, but are interpreted relative
    to ComfyUI's configured output directory rather than the process working
    directory. Resolving both paths prevents ``..`` components and existing
    symlinks from escaping the allowed root.
    """
    output_root = os.path.realpath(output_root)
    requested_directory = output_file_path
    if not os.path.isabs(requested_directory):
        requested_directory = os.path.join(output_root, requested_directory)

    candidate = os.path.realpath(
        os.path.join(requested_directory, f"{file_name}.{file_extension}")
    )
    try:
        is_within_output = os.path.commonpath((output_root, candidate)) == output_root
    except ValueError:
        # Different Windows drives and paths containing null bytes are unsafe.
        is_within_output = False

    if not is_within_output:
        raise ValueError("Saving outside the ComfyUI output directory is not allowed")

    return candidate
