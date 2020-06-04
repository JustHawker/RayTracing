#define _USE_MATH_DEFINES
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda.h"
#include "curand.h"

#include "EasyBmp\EasyBMP.h"

#include "vec.h"
#include "ray.h"
#include "sphere.h"

#include <stdio.h>
#include <time.h>

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
#define checkCurandErrors(val) check_curand( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) 
{
	if (result) 
	{
		std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
			file << ":" << line << " '" << func << "' \n";
		cudaDeviceReset();
		exit(99);
	}
}

void check_curand(curandStatus_t result, char const *const func, const char *const file, int const line)
{
	if (result != CURAND_STATUS_SUCCESS)
	{
		std::cerr << "CURAND error = " << static_cast<unsigned int>(result) << " at " <<
			file << ":" << line << " '" << func << "' \n";
		cudaDeviceReset();
		exit(101);
	}
}


__host__ __device__ vec3 reflect(const vec3& v, const vec3& n) 
{
	return v - 2 * dot(v, n)*n;
}

__host__ __device__ bool hit_sphere(ray &r, sphere &sp, float t_min, float t_max, hit_record &rec)
{
	vec3 oc = r.origin() - sp.center;

	float a = dot(r.direction(), r.direction());
	float b = 2.0f* dot(oc, r.direction());
	float c = dot(oc, oc) - sp.radius * sp.radius;

	float D = b * b - 4 * a*c;

	if(D>0)
	{
		float temp = (-b - sqrt(D)) / (2 * a);

		temp = (temp > t_min && temp < t_max)? 
			temp : (-b + sqrt(D)) / (2 * a);

		bool is_inbound = (temp > t_min && temp < t_max);

		if (is_inbound)
		{
			rec.t = temp;
			rec.p = r.point_at_parameter(rec.t);
			rec.normal = (rec.p - sp.center) / sp.radius;
			rec.reflect_coef = sp.reflect_coef;
			rec.color = sp.color;
		}
		return is_inbound;
	}
	return false;
}

__host__ __device__ bool hit_world(ray &r, sphere *spheres, int num_spheres, hit_record &rec)
{
	hit_record temp_rec;
	bool hit_anything = false;
	double closest_so_far = FLT_MAX;
	for (int i = 0; i < num_spheres; ++i)
	{
		if (hit_sphere(r, spheres[i], 0.0001, closest_so_far, temp_rec))
		{
			hit_anything = true;
			closest_so_far = temp_rec.t;
			rec = temp_rec;
		}
	}
	return hit_anything;
}

__host__ __device__ float get_illumination(hit_record &rec, sphere *spheres, int s_c, vec3 *light_sources, int lc)
{
	hit_record rec2;
	float dt3 = 0;
	for (int i = 0; i < lc; ++i)
	{
		ray surface_ray(rec.p, light_sources[i] - rec.p);
		if (!hit_world(surface_ray, spheres, s_c, rec2))
		{
			vec3 light_ray = rec.p - light_sources[i];
			light_ray.make_unit_vector();
			float dt = -dot(light_ray, rec.normal)*dot(light_ray, light_ray);

			dt = dt < 0.0f ? 0.0f : dt;
			dt = dt > 1.0f ? 1.0f : dt;

			dt3 += dt;
		}
	}
	return dt3 / (float) lc;
}

 __host__ __device__ vec3 color(ray &r, sphere *spheres, vec3 *lights, int s_c, int l_c)
{
	ray tray(r.A, r.B);
	vec3 result(0, 0, 0);
	hit_record rec;

	vec3 c[5];
	float rf[5];
	float d[5];
	int end_stage = 4;

	for (int j = 0; j < 5; ++j)
	{
		if(hit_world(tray,spheres,s_c,rec))
		{
			d[j] =  get_illumination(rec, spheres, s_c, lights, l_c);
			rf[j] = rec.reflect_coef;
			c[j] = rec.color;

			tray.A = rec.p;
			tray.B = reflect(rec.p, rec.normal);
		}
		else
		{
			end_stage = j-1;
			j = 5;
		}
	}
	for (int j = end_stage; j >= 0; --j)
		result = ((1.0f - rf[j])*c[j] + rf[j] * result)*d[j];

	return result;
}

__global__ void kernel_test1(vec3 *fb, int max_x, int max_y, sphere *spheres, vec3 *lights, int s_c, int l_c)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= max_x) || (j >= max_y)) return;

	float g = 4.0f*((float)max_x / (float)max_y);

	vec3 origin(0.0f, 0.0f, 0.0f);
	vec3 vertical(0.0f, 4.0f, 0.0f);
	vec3 horizontal(g, 0.0f, 0.0f);
	vec3 lower_left_corner(-g/2, -2.0f, -2.0f);

	float u = ((float)i) / (float)max_x;
	float v = ((float)j) / (float)max_y;

	ray r(origin, lower_left_corner + u * horizontal + v * vertical);

	vec3 c= color(r, spheres, lights, s_c, l_c);

	size_t pixel_index = j * max_x + i;
	fb[pixel_index] = c;
}


__host__ void fill_world(sphere *spheres, vec3 *lights, int s_c, int l_c)
{
	int size1 = 40*sizeof(float);
	int size = 40;
	float *t = (float *)malloc(size1);
	curandGenerator_t gen;
	checkCurandErrors(curandCreateGeneratorHost(&gen, CURAND_RNG_PSEUDO_DEFAULT));
	checkCurandErrors(curandSetPseudoRandomGeneratorSeed(gen,clock()));

	checkCurandErrors(curandGenerateUniform(gen, t, size));
	for (int i = 0; i < s_c*3; i+=3)
		spheres[(int)(i/3)].color.set(t[i], t[i + 1], t[i + 2]);

	checkCurandErrors(curandGenerateUniform(gen, t, size));
	for (int i = 0; i < s_c; ++i)
		spheres[i].reflect_coef = t[i];

	checkCurandErrors(curandGenerateUniform(gen, t, size));
	for (int i = 0; i < s_c; ++i)
		spheres[i].radius = t[i]*0.5f;

	checkCurandErrors(curandGenerateNormal(gen, t, size, -0.5f, 0.2f));
	for (int i = 0; i < s_c; ++i)
		spheres[i].center.y() = t[i];

	checkCurandErrors(curandGenerateNormal(gen, t, size, 0, 2.0f));
	for (int i = 0; i < s_c; ++i)
		spheres[i].center.x() = t[i];

	checkCurandErrors(curandGenerateNormal(gen, t, size, -3, 2.0f));
	for (int i = 0; i < s_c; ++i)
		spheres[i].center.z() = t[i];


	checkCurandErrors(curandGenerateNormal(gen, t, size, 4, 2.0f));
	for (int i = 0; i < l_c; ++i)
		lights[i].y() = t[i];

	checkCurandErrors(curandGenerateNormal(gen, t, size, 0, 10.0f));
	for (int i = 0; i < l_c; ++i)
		lights[i].x() = t[i];

	checkCurandErrors(curandGenerateNormal(gen, t, size, 0, 10.0f));
	for (int i = 0; i < l_c; ++i)
		lights[i].z() = t[i];
	free(t);
	checkCurandErrors(curandDestroyGenerator(gen));
}

 void fill_world_demo(sphere *spheres, vec3 *lights)
 {
	 lights[0].set(0, 1, 0);
	 lights[1].set(2, 2, 2);

	 float r = 1.75f;
	 int j = 0;
	 for (float i = 0; i < 2*M_PI; i+= 2*M_PI/9)
	 {
		 float x = r * cos(i);
		 float y = r * sin(i);
		 spheres[j].set(sphere(vec3(x, 0, y), 0.5f, 0.5f, vec3(1, 0, 1)));
		 j++;
		 if (j > 8) break;
	 }
	 spheres[0].color.set(1, 1, 1);
	 spheres[1].color.set(1, 1, 0);
	 spheres[2].color.set(1, 0, 1);
	 spheres[3].color.set(1, 0, 0);
	 spheres[4].color.set(0, 1, 1);
	 spheres[5].color.set(0, 1, 0);
	 spheres[6].color.set(0.5f, 0.5f, 0.5f);
	 spheres[7].color.set(0.9f, 0.5f, 0.1f);
	 spheres[8].color.set(0, 0, 0);

	 spheres[9].set(sphere(vec3(0, -500.5, -1), 500, 0.0f, vec3(0, 1, 1)));
 }

int main()
{
	cudaDeviceProp prop;
	checkCudaErrors(cudaGetDeviceProperties(&prop, 0));
	printf("Your CUDA-compatible device: %s\n", prop.name);

	char filename[256];
	int sc, lc;
	int nx, ny;	
	char d;

	printf("Enter output filename (with extention): ");
	scanf("%256s", &filename);

	printf("Enter image width [800-1920]: ");
	do{scanf("%d", &nx);} while (!(nx >= 800 && nx <= 1920));

	printf("Enter image height [600-1080]: ");
	do {scanf("%d", &ny);} while (!(ny >= 600 && ny <= 1080));

	printf("Do you want render demo scene? (y/n): ");
	do { scanf("%c", &d); } while (d != 'y' && d !='n');

	if(d=='n')
	{
		printf("How many spheres you want to generate? [5-10]: ");
		do { scanf("%d", &sc); } while (!(sc >= 5 && sc <= 10));

		printf("How many dot-lights you want to generate? [1-2]: ");
		do { scanf("%d", &lc); } while (!(lc >= 1 && lc <= 2));
	}
	else
	{
		sc = 10;
		lc = 2;
	}
	printf("Do you want to also redner scene on  CPU to see execution-time speedup? (y/n): ");
	char temp;
	do { scanf("%c", &temp); } while (temp != 'y' && temp != 'n');
	bool render_cpu = (temp=='y');


	cudaEvent_t start, stop;
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));

	int tx = 8;
	int ty = 8;

	sphere *spheres = (sphere*) malloc(sc*sizeof(sphere));
	vec3 *lights = (vec3*) malloc(lc * sizeof(vec3));

	if (d != 'n')
		fill_world_demo(spheres, lights);
	else
		fill_world(spheres, lights, sc, lc);

	sphere *spheres_gpu;
	vec3 *lights_gpu;

	checkCudaErrors(cudaMalloc((void**)&spheres_gpu, sizeof(sphere) * sc));
	checkCudaErrors(cudaMalloc((void**)&lights_gpu, sizeof(vec3) * lc));

	vec3 *framebuffer_gpu;
	vec3 *framebuffer = (vec3*)malloc(nx*ny * sizeof(vec3));

	checkCudaErrors(cudaMemcpy(spheres_gpu, spheres, sc * sizeof(sphere), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(lights_gpu, lights, lc * sizeof(vec3), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((void**)&framebuffer_gpu, nx*ny * sizeof(vec3)));
	checkCudaErrors(cudaGetLastError());

	dim3 blocks(nx / tx + 1, ny / ty + 1);
	dim3 threads(tx, ty);

	checkCudaErrors(cudaEventRecord(start));
	kernel_test1<<<blocks, threads >>>(framebuffer_gpu, nx, ny,spheres_gpu,lights_gpu,sc,lc);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaEventRecord(stop));
	checkCudaErrors(cudaEventSynchronize(stop));

	float milliseconds = 0;
	checkCudaErrors(cudaEventElapsedTime(&milliseconds, start, stop));
	milliseconds /= 1000.0f;

	printf("GPU rendering time: %.5f", milliseconds);

	checkCudaErrors(cudaFree(spheres_gpu));
	checkCudaErrors(cudaFree(lights_gpu));

	checkCudaErrors(cudaMemcpy(framebuffer, framebuffer_gpu, nx *ny * sizeof(vec3), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaFree(framebuffer_gpu));

	BMP out_file;
	out_file.SetSize(nx, ny);
	for (int i = 0; i < nx; ++i)
	{
		for (int j = 0; j < ny; ++j)
		{
			vec3 t = framebuffer[j*nx + i];
			int ir = int(255.99f*t.x());
			int ig = int(255.99f*t.y());
			int ib = int(255.99f*t.z());

			out_file(i, ny-j-1)->Red = ir;
			out_file(i, ny - j - 1)->Green = ig;
			out_file(i, ny - j - 1)->Blue = ib;
		}
	}

	out_file.WriteToFile("result.bmp");

	if (render_cpu)
	{
		BMP out_file_cpu;
		out_file_cpu.SetSize(nx, ny);


		float g = 4.0f*((float)nx / (float)ny);
		vec3 origin(0.0f, 0.0f, 0.0f);
		vec3 vertical(0.0f, 4.0f, 0.0f);
		vec3 horizontal(g, 0.0f, 0.0f);
		vec3 lower_left_corner(-g / 2, -2.0f, -2.0f);
		float elapsed = 0.0f;
		for (int i = 0; i < ny; ++i)
		{
			for (int j = 0; j < nx; ++j)
			{
				clock_t start = clock();

				vec3 col(0, 0, 0);
				float v = ((float)i) / (float)ny;
				float u = ((float)j) / (float)nx;

				ray r(origin, lower_left_corner + u * horizontal + v * vertical);

				col = color(r, spheres, lights, sc, lc);

				elapsed += difftime(clock(), start);

				int ir = int(255.99f*col[0]);
				int ig = int(255.99f*col[1]);
				int ib = int(255.99f*col[2]);

				out_file_cpu(j, ny - i - 1)->Red = ir;
				out_file_cpu(j, ny - i - 1)->Green = ig;
				out_file_cpu(j, ny - i - 1)->Blue = ib;
			}
		}
		out_file_cpu.WriteToFile("cpu_result.bmp");
		elapsed /= CLOCKS_PER_SEC;
		printf("\nCPU elapsed time: %.5f\n", elapsed);
		printf("speedup: %.5f", elapsed / milliseconds);
	}

	free(spheres);
	free(lights);
	free(framebuffer);

	return 0;
}