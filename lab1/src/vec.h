//#pragma once

#ifndef VEC_H
#define VEC_H

#include<math.h>

class vec3
{public:
	float e[3];

	__host__ __device__ vec3() {}
	__host__ __device__ vec3(float e0, float e1, float e2) {e[0] = e0; e[1] = e1; e[2] = e2;}

	__host__ __device__ float &x() { return e[0]; }
	__host__ __device__ float &y() { return e[1]; }
	__host__ __device__ float &z() { return e[2]; }

	__host__ __device__ float &r() { return e[0]; }
	__host__ __device__ float &g() { return e[1]; }
	__host__ __device__ float &b() { return e[2]; }

	__host__ __device__  const vec3& operator+() { return *this; }
	__host__ __device__ vec3 operator-() { return vec3(-e[0], -e[1], -e[2]); }
	__host__ __device__ float operator[](int i) { return e[i]; }

	__host__ __device__ vec3& operator+=(const vec3& v2);
	__host__ __device__ vec3& operator-=(const vec3& v2);
	__host__ __device__ vec3& operator*=(const vec3& v2);
	__host__ __device__ vec3& operator/=(const vec3& v2);
	__host__ __device__ vec3& operator*=(const float t);
	__host__ __device__ vec3& operator/=(const float t);

	__host__ __device__ float squared_length() {
		return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
	}
	__host__ __device__ float length() {
		return sqrt(squared_length());
	}
	__host__ __device__ void make_unit_vector();
	__host__ __device__ void set(float x, float y, float z);
};

__host__ __device__ void vec3::set(float x, float y, float z)
{
	e[0] = x;
	e[1] = y;
	e[2] = z;
}

__host__ __device__ void vec3::make_unit_vector() {
	float k = 1.0f / vec3::length();
	e[0] *= k; e[1] *= k; e[2] *= k;
}

__host__ __device__ vec3 operator+(const vec3 &v1, const vec3 &v2)
{
	return vec3(v1.e[0]+v2.e[0], v1.e[1]+v2.e[1], v1.e[2]+v2.e[2]);
}
__host__ __device__ vec3 operator-(const vec3 &v1, const vec3 &v2)
{
	return vec3(v1.e[0] - v2.e[0], v1.e[1] - v2.e[1], v1.e[2] - v2.e[2]);
}
__host__ __device__ vec3 operator*(const vec3 &v1, const vec3 &v2)
{
	return vec3(v1.e[0] * v2.e[0], v1.e[1] * v2.e[1], v1.e[2] * v2.e[2]);
}
__host__ __device__ vec3 operator/(const vec3 &v1, const vec3 &v2)
{
	return vec3(v1.e[0] / v2.e[0], v1.e[1] / v2.e[1], v1.e[2] / v2.e[2]);
}

__host__ __device__ vec3 operator*(const vec3 &v, float t)
{
	return vec3(v.e[0] * t, v.e[1] * t, v.e[2] * t);
}

__host__ __device__ vec3 operator*(const float t, const vec3 &v)
{
	return vec3(v.e[0] * t, v.e[1] * t, v.e[2] * t);
}

__host__ __device__ vec3 operator/(const vec3 &v, float t)
{
	return vec3(v.e[0] / t, v.e[1] / t, v.e[2] / t);
}

__host__ __device__ float dot(const vec3 &v1, const vec3 &v2)
{
	return v1.e[0] * v2.e[0] + v1.e[1] * v2.e[1] + v1.e[2] * v2.e[2];
}

__host__ __device__ vec3 cross(const vec3 &v1, const vec3 &v2)
{
	return vec3((v1.e[1] * v2.e[2] - v1.e[2] * v2.e[1]),
		-(v1.e[0] * v2.e[2] - v1.e[2] * v2.e[0]),
		(v1.e[0] * v2.e[1] - v1.e[1] * v2.e[0]));
}

__host__ __device__ vec3& vec3::operator+=(const vec3 &v)
{
	e[0] += v.e[0];
	e[1] += v.e[1];
	e[2] += v.e[2];
	return *this;
}

__host__ __device__ vec3& vec3::operator-=(const vec3 &v)
{
	e[0] -= v.e[0];
	e[1] -= v.e[1];
	e[2] -= v.e[2];
	return *this;
}

__host__ __device__ vec3& vec3::operator*=(const vec3 &v)
{
	e[0] *= v.e[0];
	e[1] *= v.e[1];
	e[2] *= v.e[2];
	return *this;
}

__host__ __device__ vec3& vec3::operator/=(const vec3 &v)
{
	e[0] /= v.e[0];
	e[1] /= v.e[1];
	e[2] /= v.e[2];
	return *this;
}

__host__ __device__ vec3& vec3::operator/=(const float t)
{
	float k = 1.0f / t;
	e[0] *= k;
	e[1] *= k;
	e[2] *= k;
	return *this;
}

__host__ __device__ vec3& vec3::operator*=(const float t)
{
	e[0] *= t;
	e[1] *= t;
	e[2] *= t;
	return *this;
}

__host__ __device__ vec3 unit_vector(vec3 v)
{
	return v / v.length();
}
#endif // !VEC_H