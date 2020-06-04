#pragma once

#ifndef SPHERE_H
#define SPHERE_H

#include "hit_record.h"

#include <memory>
#include <vector>

class sphere 
{
public:
	float radius;
	vec3 center;
	float reflect_coef;
	vec3 color;

	sphere(){}
	sphere(vec3 cen, float r, float reflect, vec3 col) : 
		center(cen), radius(r), reflect_coef(reflect), color(col) {};
	void set(sphere &s);
};

void sphere::set(sphere &s)
{
	radius = s.radius;
	center = s.center;
	color = s.color;
	reflect_coef = s.reflect_coef;
}
#endif // !SPHERE_H