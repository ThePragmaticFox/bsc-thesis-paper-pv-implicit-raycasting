/* 
    XAC CUDA Kernel for Nvidia IndeX
    Implementation of a general Parallel Vectors Operator formulated by Peikert & Roth
    https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.90.5182&rep=rep1&type=pdf

    Summary
    Given three float_4 IndeX volumes v, w and s, this algorithm will return the shaded curves of v∥w, where
    v.x, v.y, v.z and w.x, w.y, w.z are the two vector fields for which we want to find the parallel curves,
    s.x ∈ {0,1} is a mask intended to pre-filter the search region,
    s.y ∈ [0,1] determines the radius of the intersected geometry at the parallel curves,
    s.z, s.w ∈[0,1] help to determine the color(map) and alpha values of the final visualisation output.

    Author
    Written 2019 by Ramon Witschi, ETH Computer Science BSc, Bachelor Thesis @ CGL
*/

NV_IDX_XAC_VERSION_1_0
using namespace nv::index::xac;
using namespace nv::index::xaclib;

#include "xac-header_common.h"

/*------------------------------------------------------------------------------------------------------*/
/*--------------------------- volume sample class (where the magic happens) ----------------------------*/
/*------------------------------------------------------------------------------------------------------*/

class Volume_sample_program
{
    NV_IDX_VOLUME_SAMPLE_PROGRAM

    public:

        /*------------------------------------------------------------------------------------------------------*/
        /*------------------------------------------- instantiations -------------------------------------------*/
        /*------------------------------------------------------------------------------------------------------*/

        const unsigned int fidx = 0u; // default sampling field index parameter

        const SVW_Sparse_Volumes svw_svols = {
            state.scene.access<Sparse_volume>(SLOT_vector_field_s),
            state.scene.access<Sparse_volume>(SLOT_vector_field_v),
            state.scene.access<Sparse_volume>(SLOT_vector_field_w)
        };

        const SVW_Sparse_Volume_Sampler svw_sampler = {
            svw_svols.s.generate_sampler<float4, NEAREST_FILTER, INTERP>(fidx),
            svw_svols.s.generate_sampler<float4, TRILINEAR_FILTER, INTERP>(fidx),
            svw_svols.s.generate_sampler<float4, TRICUBIC_FILTER, INTERP>(fidx),
            svw_svols.v.generate_sampler<float4, NEAREST_FILTER, INTERP>(fidx),
            svw_svols.v.generate_sampler<float4, TRILINEAR_FILTER, INTERP>(fidx),
            svw_svols.v.generate_sampler<float4, TRICUBIC_FILTER, INTERP>(fidx),
            svw_svols.w.generate_sampler<float4, NEAREST_FILTER, INTERP>(fidx),
            svw_svols.w.generate_sampler<float4, TRILINEAR_FILTER, INTERP>(fidx),
            svw_svols.w.generate_sampler<float4, TRICUBIC_FILTER, INTERP>(fidx),
        };

        /*------------------------------------------------------------------------------------------------------*/
        /*--------------------------------------- adjustable parameters ----------------------------------------*/
        /*------------------------------------------------------------------------------------------------------*/

        float3 color = make_float3(1.0f, 1.0f, 1.0f);

        float __DX = 1.0f;
        float __DY = 1.0f;
        float __DZ = 1.0f;

        uint __DATASET = 0;
        uint __S_FILTER = 1;
        uint __ADAPTIVE = 1;
        uint __MAX_ITERS = 3;

        float __STEP_SIZE = 1.0f;

        float __CMAP_MIN = 0.0f;
        float __CMAP_MAX = 1.0f;

        float __RADIUS_MIN = 1.0f;
        float __RADIUS_MAX = 0.0f;

        float __ALPHA_MIN = 1.0f;
        float __ALPHA_MAX = 0.0f;

        /*------------------------------------------------------------------------------------------------------*/
        /*--------------------------------------------- constants ----------------------------------------------*/
        /*------------------------------------------------------------------------------------------------------*/

        const float __EPSILON_DET = 0.0f;
        const float __EPSILON_CHANGE = 0.0f;
        const float __DISTANCE_THRESHOLD = 1.0f;

        const float __EPSILON_INNER_ZERO = 0.0f;//1E-3f;
        const float __EPSILON_OUTER_ZERO = 1E-2f;

        const float __EPSILON_TRUST_REGION = 1.0f;
        const float __EPSILON_TRUST_REGION_TRIGGER = 1E-5f;

        const float __SPHERE_THICKNESS_TRESH = 1E-3f;

        /*------------------------------------------------------------------------------------------------------*/
        /*----------------------------------- class initialization function ------------------------------------*/
        /*------------------------------------------------------------------------------------------------------*/

        NV_IDX_DEVICE_INLINE_MEMBER
        void initialize()
        {

            /*-------------------------- setup IndeX buffers for adjustable variables ---------------------------*/

            __DATASET =                         *state.bind_parameter_buffer<uint>(1);
            __S_FILTER =                        *state.bind_parameter_buffer<uint>(2);

            __ADAPTIVE =                        *state.bind_parameter_buffer<uint>(3);
            __MAX_ITERS =                       *state.bind_parameter_buffer<uint>(4);
            __STEP_SIZE =                       *state.bind_parameter_buffer<float>(5);

            __CMAP_MIN =                        *state.bind_parameter_buffer<float>(6);
            __CMAP_MAX =                        *state.bind_parameter_buffer<float>(7);

            __RADIUS_MIN =                      *state.bind_parameter_buffer<float>(8);
            __RADIUS_MAX =                      *state.bind_parameter_buffer<float>(9);

            __ALPHA_MIN =                       *state.bind_parameter_buffer<float>(10);
            __ALPHA_MAX =                       *state.bind_parameter_buffer<float>(11);

            switch(__DATASET)
            {
                case 1 : // Moving Center

                    __DX = 0.031496062992125984;
                    __DY = 0.031496062992125984;
                    __DZ = 0.031496062992125984;

                    break;

                case 2 : // Stuart Vortex

                    __DX = 0.06299212598425197;
                    __DY = 0.031496062992125984;
                    __DZ = 0.04947390005511811;

                    break;

                case 3 : // Tornado

                    __DX = 0.15748031496062992;
                    __DY = 0.15748031496062992;
                    __DZ = 0.15748031496062992;

                    break;

                case 4 : // Borromean Rings

                    __DX = 0.049087401574803145;
                    __DY = 0.049087401574803145;
                    __DZ = 0.049087401574803145;

                    break;

                case 5 : // Delta Wing

                    __DX = 0.10040160642570281;
                    __DY = 0.10161290322580645;
                    __DZ = 0.10101010101010101;

                    break;

                case 6 : // Cylinder_Gauss3_Subset

                    __DX = 0.012519561815336464;
                    __DY = 0.012658227848101266;
                    __DZ = 0.010000000000000002;

                    break;

                case 7 : // Swirling Jet

                    __DX = 1.0;
                    __DY = 1.0;
                    __DZ = 1.0;

                    break;
            }
        }

        /*------------------------------------------------------------------------------------------------------*/
        /*--------------------------------------- class execute function ---------------------------------------*/
        /*------------------------------------------------------------------------------------------------------*/

        NV_IDX_DEVICE_INLINE_MEMBER
        int execute(const Sample_info_self& sample_info, Sample_output& sample_output)
        {

            /*-------------------------------------------- initialize -------------------------------------------*/

            float4 color_output;
            bool colormap_flag = true;
            const Sparse_volume& sparse_volume = state.self;

            const float3 sample_position = sample_info.sample_position_object_space;
            const float4 s = svw_sampler.s_nearest.fetch_sample(sample_position);
            const Colormap colormap = sparse_volume.get_colormap();

            /*---------------------------------------- scalar filter field s -------------------------------------*/

            if ( ( ! __S_FILTER ) || ( __S_FILTER && s.x ) )
            {
                /*------------------------------ parallel vectors v || w <=> v x w = 0 -------------------------------*/

                bool has_converged = true;
                float3 root = sample_position;

                if ( __MAX_ITERS != 0 )
                {
                    const uint max_iterations = min(100, __MAX_ITERS);
                    has_converged = find_root(
                        sample_position,
                        root,
                        max_iterations,
                        svw_sampler,
                        __ADAPTIVE,
                        __STEP_SIZE,
                        __EPSILON_DET,
                        __EPSILON_CHANGE,
                        __DISTANCE_THRESHOLD,
                        __EPSILON_INNER_ZERO,
                        __EPSILON_OUTER_ZERO,
                        __EPSILON_TRUST_REGION,
                        __EPSILON_TRUST_REGION_TRIGGER
                    );
                }

                const float4 s_root = svw_sampler.s_nearest.fetch_sample(root);

                if ( has_converged && ( ( ! __S_FILTER ) || ( __S_FILTER && s_root.x ) ) )
                {

                    /*------------------------------------------- intersection -------------------------------------------*/

                    const Ray ray = {sample_info.ray_t, sample_info.ray_origin, normalize(sample_info.ray_direction)};
                    const Sphere sphere = {__RADIUS_MIN + s_root.y*__RADIUS_MAX, root};
                    const Intersect intersect = get_intersect_sphere(ray, sphere, __SPHERE_THICKNESS_TRESH);

                    if ( intersect.is_intersected )
                    {

                        /*------------------------------------------- shading output -----------------------------------------*/

                        const float alpha = __ALPHA_MIN + s_root.w*__ALPHA_MAX;
                        const float3 shading_position = ray.origin + intersect.t*ray.direction;
                        const float3 normal = shading_position - root;

                        const float3 cshading = get_shading_color(shading_position, normal, color);
                        color_output = make_float4(cshading.x, cshading.y, cshading.z, alpha);

                        colormap_flag = false;

                    }
                }
            }

            if ( colormap_flag )
            {
                /*------------------------------------------ colormap output -----------------------------------------*/

				/* The following contains tests not suitable for all datasets */

                const float3 v = get_v_trilinear(sample_position, svw_sampler);
                const float3 w = get_w_trilinear(sample_position, svw_sampler);
                const float velocity_magnitude = get_3d_norm(v);
                const float acceleration_magnitude = get_3d_norm(w);

                const Mat3f Jacobian = get_v_jacobian_numerical(sample_position, svw_sampler, __DX, __DY, __DZ);
                const float vorticity_magnitude = get_vorticity_magnitude(sample_position, svw_sampler, Jacobian);

                const Eigenvalues3f eigs = get_3d_eigenvalues(Jacobian);

                const float cand1 = get_3d_norm(eigs.ev1_real*v - w);
                const float cand2 = get_3d_norm(eigs.ev2_real*v - w);
                const float cand3 = get_3d_norm(eigs.ev3_real*v - w);

                uint index = 1;
                float ev1 = eigs.ev2_real;
                float ev2 = eigs.ev3_real;
                if ( cand2 < cand1 && cand2 < cand3 )
                {
                    index = 2;
                    ev1 = eigs.ev1_real;
                    ev2 = eigs.ev3_real;
                }
                if ( cand3 < cand1 && cand3 < cand2 )
                {
                    index = 3;
                    ev1 = eigs.ev1_real;
                    ev2 = eigs.ev2_real;
                }

                const float eps = 0.1;

                if ( ( ev1 < -eps && ev2 > eps) || ( ev1 > eps && ev2 < -eps ) )
                {
                    color_output = make_float4(0.0f, 0.0f, abs(ev1) + abs(ev2), __ALPHA_MAX);
                }
                else
                {
                    color_output = colormap.lookup(vorticity_magnitude);
                }

                //if ( __CMAP_MIN > cmap_transfer_function || cmap_transfer_function > __CMAP_MAX ) return NV_IDX_PROG_DISCARD_SAMPLE;

                // Swirling Jet only
                const float x = sample_position.x;
                const float y = sample_position.y;
                const float z = sample_position.z;
                const float xd = 100.5f - x;
                const float zd = 100.5f - z;
                const float xxzz = sqrt(xd*xd + zd*zd);
                if (y > 218 && xxzz > 90) return NV_IDX_PROG_DISCARD_SAMPLE;

                //const float cmap_transfer_function = velocity_magnitude;
                //const float cmap_transfer_function = acceleration_magnitude;

                const Mat3f Jacobian = get_v_jacobian_numerical(sample_position, svw_sampler, __DX, __DY, __DZ);
                const float vorticity_magnitude = get_vorticity_magnitude(sample_position, svw_sampler, Jacobian);
                const float cmap_transfer_function = vorticity_magnitude;
                if ( __CMAP_MIN > cmap_transfer_function || cmap_transfer_function > __CMAP_MAX ) return NV_IDX_PROG_DISCARD_SAMPLE;
                color_output = colormap.lookup(cmap_transfer_function);
            }

            /*-------------------------------------------- return RGBA -------------------------------------------*/

            sample_output.set_color(clamp(color_output, 0.0f, 1.0f));
            return NV_IDX_PROG_OK;

        }

}; // class Volume_sample_program