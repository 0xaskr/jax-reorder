"""Independent small-domain enumeration oracle; never imported by production code."""

from collections import Counter


def enumerate_blocked(size, block, programs, mapping, mask=lambda address: True):
    """Count writes directly from program/lane pairs, without symbolic-domain code."""
    if programs * block > 8192:
        raise ValueError("the oracle is deliberately restricted to small domains")
    writes = Counter()
    outside = []
    for program in range(programs):
        for lane in range(block):
            address = block * mapping(program) + lane
            if mask(address):
                if 0 <= address < size:
                    writes[address] += 1
                else:
                    outside.append(address)
    complete = not outside and all(writes[i] == 1 for i in range(size))
    return complete, writes, outside


def main():
    count = 0
    for size in range(1, 65):
        for block in (1, 2, 4, 8, 16):
            programs = (size + block - 1) // block
            for reverse in (False, True):

                def mapping(p, programs=programs, reverse=reverse):
                    return programs - 1 - p if reverse else p

                good, _, _ = enumerate_blocked(
                    size, block, programs, mapping, lambda i, size=size: i < size
                )
                assert good
                bad, _, _ = enumerate_blocked(
                    size, block, programs, mapping, lambda i, size=size: i < size - 1
                )
                assert not bad
                count += 2
    print(f"Pallas small-domain oracle: {count} coverage/gap cases passed")


if __name__ == "__main__":
    main()
