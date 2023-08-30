def generatePayoffs(i, j, values):
    a, b, c, d, errorRange, n = values

    x, y = 0, 0
    if i <= j:
        big, small = j/n, i/n
    else:
        big, small = i/n, j/n

    base = 0
    if 0 < small - errorRange:
        x, y = x + d * (small - errorRange), y + d * (small - errorRange)
        base = small - errorRange

    # This block continues from one player possibly punishing up to when the other player is the only one punishing
    if small + errorRange < big - errorRange:  # No overlap between players
        if small + errorRange < 0:
            top = 0
        else:
            top = small + errorRange

        if big - errorRange > 1:  # Very large big
            end = 1
        elif big - errorRange < 0:  # Very tiny big
            end = 0
        else:
            end = big - errorRange

        if top <= end:  # Normal case
            x += b * oneDistYes(errorRange, small, base, top) + d * oneDistNo(errorRange, small, base, top)
            y += c * oneDistYes(errorRange, small, base, top) + d * oneDistNo(errorRange, small, base, top)
            # Possibility of one player punishing

            x += b * (end - top)
            y += c * (end - top)
            # One player always punishes
        else:
            x += b * oneDistYes(errorRange, small, base, end) + d * oneDistNo(errorRange, small, base, end)
            y += c * oneDistYes(errorRange, small, base, end) + d * oneDistNo(errorRange, small, base, end)

        base = end

    else:  # Overlap
        if big - errorRange < 0:  # Tiny Big
            oneTop = 0
        else:
            oneTop = big - errorRange

        if small + errorRange > 1:  # Large small
            twoTop = 1
        else:
            twoTop = small + errorRange

        x += b * oneDistYes(errorRange, small, base, oneTop) + d * oneDistNo(errorRange, small, base, oneTop)
        y += c * oneDistYes(errorRange, small, base, oneTop) + d * oneDistNo(errorRange, small, base, oneTop)
        # Up until the second player is an option

        x += a * twoDistYes(errorRange, small, big, oneTop, twoTop) + \
             b * twoDistMaybe(errorRange, small, big, oneTop, twoTop) + \
             c * twoDistMaybe(errorRange, big, small, oneTop, twoTop) + \
             d * twoDistNo(errorRange, small, big, oneTop, twoTop)
        y += a * twoDistYes(errorRange, small, big, oneTop, twoTop) + \
             c * twoDistMaybe(errorRange, small, big, oneTop, twoTop) + \
             b * twoDistMaybe(errorRange, big, small, oneTop, twoTop) + \
             d * twoDistNo(errorRange, small, big, oneTop, twoTop)
        # overlap between two players

        base = twoTop

    if big + errorRange < 1:
        x += a * oneDistYes(errorRange, big, base, big + errorRange) + b * oneDistNo(errorRange, big, base, big + errorRange)
        y += a * oneDistYes(errorRange, big, base, big + errorRange) + c * oneDistNo(errorRange, big, base, big + errorRange)
        # One player always punishes, the other sometimes punishes

        x += a * (1 - (big + errorRange))
        y += a * (1 - (big + errorRange))
    else:
        x += a * oneDistYes(errorRange, big, base, 1) + b * oneDistNo(errorRange, big, base, 1)
        y += a * oneDistYes(errorRange, big, base, 1) + c * oneDistNo(errorRange, big, base, 1)
        # One player always punishes, the other sometimes punishes

    if small == i/n:
        return x, y
    else:
        return y, x


def oneDistYes(e, center, bottom, top):
    return integ(e-center, 1, 0, bottom, top, 2*e)


def oneDistNo(e, center, bottom, top):
    return integ(center + e, -1, 0, bottom, top, 2*e)


def twoDistYes(e, center1, center2, bottom, top):
    deg1 = center1 * center2 - e * center1 - e * center2 + pow(e, 2)
    deg2 = 2 * e - center1 - center2
    deg3 = 1

    return integ(deg1, deg2, deg3, bottom, top, 4*pow(e, 2))


def twoDistMaybe(e, centerYes, centerNo, bottom, top):
    deg1 = e * centerYes + pow(e, 2) - centerYes * centerNo - e * centerNo
    deg2 = centerYes + centerNo
    deg3 = -1

    return integ(deg1, deg2, deg3, bottom, top, 4*pow(e, 2))


def twoDistNo(e, center1, center2, bottom, top):
    deg1 = pow(e, 2) + center1 * e + center2 * e + center1 * center2
    deg2 = - 2 * e - center1 - center2
    deg3 = 1

    return integ(deg1, deg2, deg3, bottom, top, 4*pow(e, 2))


def integ(deg1, deg2, deg3, bottom, top, divide):
    total = deg1 * (top - bottom) + 1/2 * deg2 * (pow(top, 2) - pow(bottom, 2)) + 1 / 3 * deg3 * (pow(top, 3) - pow(bottom, 3))
    return total / divide
