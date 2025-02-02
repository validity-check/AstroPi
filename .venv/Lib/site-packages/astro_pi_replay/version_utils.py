def compare_semver(version1: str, version2: str, ignore_patch=False) -> int:
    """
    Compares the two semver strings.
    ignore_patch: Whether to only compare the major and
    minor versions.

    Returns 1 when version1 > version2
    Returns 0 when version1 == version2
    Returns -1 when version1 < version2
    """

    version1_split: list[str] = version1.split(".")
    version2_split: list[str] = version2.split(".")

    length: int = len(version1_split)
    if ignore_patch and length == 3:
        length = 2
    for i in range(length):
        v1: int = int(version1_split[i])
        v2: int = int(version2_split[i])
        if v1 < v2:
            return -1
        if v1 > v2:
            return 1
    return 0


def decrement_semver(version: str):
    """
    version must be a semantic version string
    of the form x.y.z where x y and z are non-negative
    """
    split_version: list[int] = [int(v) for v in version.split(".")]
    if split_version[2] == 0:
        if split_version[1] == 0:
            if split_version[0] == 0:
                raise ValueError("Cannot decrement 0.0.0")
            else:
                return f"{split_version[0]-1}.{split_version[1]}.{split_version[2]}"
        else:
            return f"{split_version[0]}.{split_version[1]-1}.{split_version[2]}"
    else:
        return f"{split_version[0]}.{split_version[1]}.{split_version[2]-1}"
