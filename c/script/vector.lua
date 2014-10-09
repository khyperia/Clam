function addvec(left, right)
    return {left[1] + right[1], left[2] + right[2], left[3] + right[3]}
end

function subvec(left, right)
    return {left[1] - right[1], left[2] - right[2], left[3] - right[3]}
end

function mulvec(left, right)
    return {left[1] * right, left[2] * right, left[3] * right}
end

function cross(v1, v2, vR)
    return { ( (v1[2] * v2[3]) - (v1[3] * v2[2]) ),
            -( (v1[1] * v2[3]) - (v1[3] * v2[1]) ),
             ( (v1[1] * v2[2]) - (v1[2] * v2[1]) ) }
end

function normalize(v)
    length = v[1] * v[1] + v[2] * v[2] + v[3] * v[3]
    return mulvec(v, 1 / math.sqrt(length))
end

function rotate(v, axis, amount)
    return addvec(mulvec(cross(axis, v), amount), v)
end

function catmullRomF(p0, p1, p2, p3, t)
    local t2 = t * t;
    local t3 = t2 * t;

    return ((2 * p1) +
        (-p0 + p2) * t +
        (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 +
        (-p0 + 3 * p1 - 3 * p2 + p3) * t3) / 2
end

function catmullRom(p0, p1, p2, p3, t)
    local t2 = t * t
    local t3 = t2 * t

    local p2r = addvec(
            subvec(mulvec(p0, 2), mulvec(p1, 5)),
            subvec(mulvec(p2, 4), p3))
    local p3r = subvec(
            addvec(mulvec(p0, -1), mulvec(p1, 3)),
            addvec(mulvec(p2, 3), mulvec(p3, -1)))

    return mulvec(
        addvec(
            addvec(
                mulvec(p1, 2),
                mulvec(subvec(p2, p0), t)),
            addvec(
                mulvec(p2r, t2),
                mulvec(p3r, t3)
            )),
        1/2.0)
end
