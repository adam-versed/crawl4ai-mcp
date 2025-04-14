import re


def validate_and_normalize_url(url: str) -> str | None:
    """Validate and normalize a URL.

    Args:
        url: The URL string to validate.

    Returns:
        The normalized URL with https scheme if valid, otherwise None.
    """
    # Simple validation for domains/subdomains with http(s)
    # Allows for optional paths
    url_pattern = re.compile(
        r"^(?:https?://)?(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|"  # domain...
        r"localhost|"  # localhost...
        r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"  # ...or ip
        r"(?::\d+)?"  # optional port
        r"(?:/?|[/?]\S+)$",
        re.IGNORECASE,
    )

    if not url_pattern.match(url):
        return None

    # Add https:// if missing
    if not url.startswith("http://") and not url.startswith("https://"):
        url = f"https://{url}"

    return url
