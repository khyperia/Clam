function addvec(left, right)
    return {left[1] + right[1], left[2] + right[2], left[3] + right[3]}
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
