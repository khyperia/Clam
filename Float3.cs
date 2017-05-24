using System;

namespace Clam4
{
    internal class Float3
    {
        public float X { get; }
        public float Y { get; }
        public float Z { get; }

        public Float3(float x, float y, float z)
        {
            X = x;
            Y = y;
            Z = z;
        }

        public static Float3 operator +(Float3 left, Float3 right) =>
            new Float3(left.X + right.X, left.Y + right.Y, left.Z + right.Z);
        public static Float3 operator -(Float3 left, Float3 right) =>
            new Float3(left.X - right.X, left.Y - right.Y, left.Z - right.Z);
        public static Float3 operator *(Float3 left, float right) =>
            new Float3(left.X * right, left.Y * right, left.Z * right);
        public static Float3 Cross(Float3 left, Float3 right) =>
            new Float3(
                left.Y * right.Z - left.Z * right.Y,
                left.Z * right.X - left.X * right.Z,
                left.X * right.Y - left.Y * right.X);
        public float Length2 => X * X + Y * Y + Z * Z;
        public float Length => (float)Math.Sqrt(Length2);
        public Float3 Normalized => this * (1 / Length);
    }
}