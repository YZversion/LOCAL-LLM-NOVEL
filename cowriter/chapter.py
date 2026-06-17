def max_chapter_for_target(target_chapter: int) -> int:
    """续写第 N 章时，最多可见 N-1 章的信息（防未来泄漏）。"""
    return target_chapter - 1
