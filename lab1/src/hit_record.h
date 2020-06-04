#pragma once

#ifndef HIT_RECORD_H
#define HIT_RECORD_H

#include "ray.h"

struct hit_record
{
	float t;
	vec3 p;
	vec3 normal;
	float reflect_coef;
	vec3 color;
};

#endif
