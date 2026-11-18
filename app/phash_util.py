import imagehash


def compute_phash(img):
    return str(imagehash.phash(img))


def hamming(hash_a: str, hash_b: str) -> int:
    return imagehash.hex_to_hash(hash_a) - imagehash.hex_to_hash(hash_b)
