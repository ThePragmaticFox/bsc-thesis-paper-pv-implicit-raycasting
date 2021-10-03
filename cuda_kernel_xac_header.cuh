/* 
    XAC CUDA Kernel Header File for parallel vectors operator implemented for Nvidia IndeX
    Written 2019 by Ramon Witschi, ETH Computer Science BSc, Bachelor Thesis @ CGL
    See cuda_kernel_xac_parallel_vectors_operator.cuh for full disclaimer
*/

NV_IDX_XAC_VERSION_1_0
using namespace nv::index::xac;
using namespace nv::index::xaclib;

#define SLOT_vector_field_s 0
#define SLOT_vector_field_v 1
#define SLOT_vector_field_w 2

#define NEAREST_FILTER nv::index::xac::Volume_filter_mode::NEAREST
#define TRILINEAR_FILTER nv::index::xac::Volume_filter_mode::TRILINEAR
#define TRICUBIC_FILTER nv::index::xac::Volume_filter_mode::TRICUBIC_BSPLINE
#define INTERP nv::index::xac::Volume_classification_mode::POST_INTERPOLATION


/*------------------------------------------------------------------------------------------------------*/
/*----------------------------------- definition of useful structs -------------------------------------*/
/*------------------------------------------------------------------------------------------------------*/


struct SVW_Sparse_Volumes
{
    const nv::index::xac::Sparse_volume& s;
    const nv::index::xac::Sparse_volume& v;
    const nv::index::xac::Sparse_volume& w;
};


struct SVW_Sparse_Volume_Sampler
{
    const Sparse_volume_sampler<float4, NEAREST_FILTER, INTERP>& s_nearest;
    const Sparse_volume_sampler<float4, TRILINEAR_FILTER, INTERP>& s_trilinear;
    const Sparse_volume_sampler<float4, TRICUBIC_FILTER, INTERP>& s_tricubic;
    const Sparse_volume_sampler<float4, NEAREST_FILTER, INTERP>& v_nearest;
    const Sparse_volume_sampler<float4, TRILINEAR_FILTER, INTERP>& v_trilinear;
    const Sparse_volume_sampler<float4, TRICUBIC_FILTER, INTERP>& v_tricubic;
    const Sparse_volume_sampler<float4, NEAREST_FILTER, INTERP>& w_nearest;
    const Sparse_volume_sampler<float4, TRILINEAR_FILTER, INTERP>& w_trilinear;
    const Sparse_volume_sampler<float4, TRICUBIC_FILTER, INTERP>& w_tricubic;
};


struct Mat2f
{
    float _11; float _12;
    float _21; float _22;
};


struct Mat3f
{
    float _11; float _12; float _13;
    float _21; float _22; float _23;
    float _31; float _32; float _33;
};


struct Ray
{
    float t;
    float3 origin;
    float3 direction;
};


struct Sphere
{
    float radius;
    float3 center;
};


struct Intersect
{
    bool is_intersected;
    float t;
};


struct Eigenvalues3f
{
    float ev1_real;
    float ev2_real;
    float ev2_complex;
    float ev3_real;
    float ev3_complex;
};

/*------------------------------------------------------------------------------------------------------*/
/*------------------------------------------- 1d primitives --------------------------------------------*/
/*------------------------------------------------------------------------------------------------------*/


NV_IDX_DEVICE_INLINE_MEMBER
float get_pow2(float a)
{
    return a*a;
}


/*------------------------------------------------------------------------------------------------------*/
/*------------------------------------------- 2d primitives --------------------------------------------*/
/*------------------------------------------------------------------------------------------------------*/


NV_IDX_DEVICE_INLINE_MEMBER
Mat2f get_2d_mat_mat_mul(const Mat2f& A, const Mat2f& B)
{
    return Mat2f{
        A._11*B._11 + A._12*B._21, A._11*B._12 + A._12*B._22,
        A._21*B._11 + A._22*B._21, A._21*B._12 + A._22*B._22
    };
}


NV_IDX_DEVICE_INLINE_MEMBER
float2 get_2d_linear_equation_solution(const Mat2f& A, const float2& b, const float determinant)
{
    const float det_inv = 1.0f / determinant;

    const Mat2f A_inv = Mat2f{
        det_inv*A._22, -det_inv*A._12,
        -det_inv*A._21, det_inv*A._11
    };

    return make_float2(
        A_inv._11*b.x + A_inv._12*b.y,
        A_inv._21*b.x + A_inv._22*b.y
    );
}


NV_IDX_DEVICE_INLINE_MEMBER
Mat2f get_2d_pseudo_inverse(const Mat2f& A)
{
    // based on Martynas Sabaliauskas, Robust algorithm for $2 \times 2$ SVD, URL (version: 2017-04-13): https://scicomp.stackexchange.com/q/18766

    float s[2], u[4], v[4];
    float a[4] = { A._11, A._12, A._21, A._22 };

    s[0] = (sqrt(get_pow2(a[0] - a[3]) + get_pow2(a[1] + a[2])) + sqrt(get_pow2(a[0] + a[3]) + get_pow2(a[1] - a[2]))) / 2;
    s[1] = abs(s[0] - sqrt(get_pow2(a[0] - a[3]) + get_pow2(a[1] + a[2])));

    v[2] = (s[0] > s[1]) ? sin((atan2(2 * (a[0] * a[1] + a[2] * a[3]), a[0] * a[0] - a[1] * a[1] + a[2] * a[2] - a[3] * a[3])) / 2) : 0;
    v[0] = sqrt(1 - v[2] * v[2]);
    v[1] = -v[2];
    v[3] = v[0];

    u[0] = (s[0] != 0) ? (a[0] * v[0] + a[1] * v[2]) / s[0] : 1;
    u[2] = (s[0] != 0) ? (a[2] * v[0] + a[3] * v[2]) / s[0] : 0;
    u[1] = (s[1] != 0) ? (a[0] * v[1] + a[1] * v[3]) / s[1] : -u[2];
    u[3] = (s[1] != 0) ? (a[2] * v[1] + a[3] * v[3]) / s[1] : u[0];

    s[0] = s[0] < 1E-10 ? 0 : 1.0 / s[0];
    s[1] = s[1] < 1E-10 ? 0 : 1.0 / s[1];

    Mat2f U = Mat2f{v[0], v[1], v[2], v[3]};
    Mat2f V = Mat2f{s[0], 0, 0, s[1]};
    Mat2f W = get_2d_mat_mat_mul(U, V);
    Mat2f Z = Mat2f{u[0], u[2], u[1], u[3]};

    Mat2f A_inv = get_2d_mat_mat_mul(W, Z);

    return A_inv;
}


/*------------------------------------------------------------------------------------------------------*/
/*------------------------------------------- 3d primitives --------------------------------------------*/
/*------------------------------------------------------------------------------------------------------*/


NV_IDX_DEVICE_INLINE_MEMBER
float3 get_floor3(const float3& a)
{
    return make_float3(floor(a.x), floor(a.y), floor(a.z));
}


NV_IDX_DEVICE_INLINE_MEMBER
float get_3d_norm(const float3& a)
{
    return sqrt(dot(a, a));
}


NV_IDX_DEVICE_INLINE_MEMBER
float get_3d_determinant(const Mat3f& m)
{
    return m._11*( m._22*m._33 - m._23*m._32 )
            + m._12*( m._23*m._31 - m._21*m._33 )
            + m._13*( m._21*m._32 - m._22*m._31 );
}


NV_IDX_DEVICE_INLINE_MEMBER
Mat3f get_3d_inverse(const Mat3f& m, const float determinant)
{
    const float a = 1.0f / determinant;
    return Mat3f{
        a*( m._22*m._33 - m._32*m._23 ), a*( m._13*m._32 - m._33*m._12 ), a*( m._12*m._23 - m._22*m._13 ),
        a*( m._23*m._31 - m._33*m._21 ), a*( m._11*m._33 - m._31*m._13 ), a*( m._13*m._21 - m._23*m._11 ),
        a*( m._21*m._32 - m._31*m._22 ), a*( m._12*m._31 - m._32*m._11 ), a*( m._11*m._22 - m._21*m._12 )
    };
}


NV_IDX_DEVICE_INLINE_MEMBER
float3 get_3d_mat_vec_product(const Mat3f& A, const float3& b)
{
    return make_float3(
        b.x*A._11 + b.y*A._12 + b.z*A._13,
        b.x*A._21 + b.y*A._22 + b.z*A._23,
        b.x*A._31 + b.y*A._32 + b.z*A._33
    );
}


/*------------------------------------------------------------------------------------------------------*/
/*----------------------------------------- sampling functions -----------------------------------------*/
/*------------------------------------------------------------------------------------------------------*/


NV_IDX_DEVICE_INLINE_MEMBER
float3 get_v_trilinear(const float3& sample_position, const SVW_Sparse_Volume_Sampler& svw_sampler)
{
    const float4 v = svw_sampler.v_trilinear.fetch_sample(sample_position);
    return make_float3(v.x, v.y, v.z);
}


NV_IDX_DEVICE_INLINE_MEMBER
float3 get_w_trilinear(const float3& sample_position, const SVW_Sparse_Volume_Sampler& svw_sampler)
{
    const float4 w = svw_sampler.w_trilinear.fetch_sample(sample_position);
    return make_float3(w.x, w.y, w.z);
}


NV_IDX_DEVICE_INLINE_MEMBER
float3 get_vxw(const float3& sample_position, const SVW_Sparse_Volume_Sampler& svw_sampler)
{
    //float4 v4 = svw_sampler.v_nearest.fetch_sample(sample_position);
    float4 v4 = svw_sampler.v_trilinear.fetch_sample(sample_position);
    //float4 v4 = svw_sampler.v_tricubic.fetch_sample(sample_position);

    //float4 w4 = svw_sampler.w_nearest.fetch_sample(sample_position);
    float4 w4 = svw_sampler.w_trilinear.fetch_sample(sample_position);
    //float4 w4 = svw_sampler.w_tricubic.fetch_sample(sample_position);

    return cross(make_float3(v4.x, v4.y, v4.z), make_float3(w4.x, w4.y, w4.z));
}


NV_IDX_DEVICE_INLINE_MEMBER
Mat3f get_vxw_jacobian(const float3& sample_position, const SVW_Sparse_Volume_Sampler& svw_sampler)
{
    const float3 ex = make_float3(1.0f, 0.0f, 0.0f);
    const float3 ey = make_float3(0.0f, 1.0f, 0.0f);
    const float3 ez = make_float3(0.0f, 0.0f, 1.0f);

    const float3 floored_sample_position = get_floor3(sample_position);

    const float3 v000 = get_v_trilinear(floored_sample_position,                  svw_sampler);
    const float3 v001 = get_v_trilinear(floored_sample_position + ez,             svw_sampler);
    const float3 v010 = get_v_trilinear(floored_sample_position + ey,             svw_sampler);
    const float3 v011 = get_v_trilinear(floored_sample_position + ey + ez,        svw_sampler);
    const float3 v100 = get_v_trilinear(floored_sample_position + ex,             svw_sampler);
    const float3 v101 = get_v_trilinear(floored_sample_position + ex + ez,        svw_sampler);
    const float3 v110 = get_v_trilinear(floored_sample_position + ex + ey,        svw_sampler);
    const float3 v111 = get_v_trilinear(floored_sample_position + ex + ey + ez,   svw_sampler);

    const float3 w000 = get_w_trilinear(floored_sample_position,                  svw_sampler);
    const float3 w001 = get_w_trilinear(floored_sample_position + ez,             svw_sampler);
    const float3 w010 = get_w_trilinear(floored_sample_position + ey,             svw_sampler);
    const float3 w011 = get_w_trilinear(floored_sample_position + ey + ez,        svw_sampler);
    const float3 w100 = get_w_trilinear(floored_sample_position + ex,             svw_sampler);
    const float3 w101 = get_w_trilinear(floored_sample_position + ex + ez,        svw_sampler);
    const float3 w110 = get_w_trilinear(floored_sample_position + ex + ey,        svw_sampler);
    const float3 w111 = get_w_trilinear(floored_sample_position + ex + ey + ez,   svw_sampler);

    const float x = sample_position.x - floored_sample_position.x;
    const float y = sample_position.y - floored_sample_position.y;
    const float z = sample_position.z - floored_sample_position.z;

    float3 vec1 = lerp(lerp(w001 - w000, w101 - w100, x), lerp(w011 - w010, w111 - w110, x), y);
    float3 vec2 = lerp(lerp(v001 - v000, v101 - v100, x), lerp(v011 - v010, v111 - v110, x), y);
    float3 vec3 = lerp(lerp(w010 - w000, w110 - w100, x), lerp(w011 - w001, w111 - w101, x), z);
    float3 vec4 = lerp(lerp(v010 - v000, v110 - v100, x), lerp(v011 - v001, v111 - v101, x), z);
    float3 vec5 = lerp(lerp(w100 - w000, w110 - w010, y), lerp(w101 - w001, w111 - w011, y), z);
    float3 vec6 = lerp(lerp(v100 - v000, v110 - v010, y), lerp(v101 - v001, v111 - v011, y), z);
    float3 vec7 = lerp(lerp(lerp(v000, v100, x), lerp(v010, v110, x), y), lerp(lerp(v001, v101, x), lerp(v011, v111, x), y), z);
    float3 vec8 = lerp(lerp(lerp(w000, w100, x), lerp(w010, w110, x), y), lerp(lerp(w001, w101, x), lerp(w011, w111, x), y), z);

    float3 J1 = cross(vec6, vec8) + cross(vec7, vec5);
    float3 J2 = -cross(vec8, vec4) - cross(vec3, vec7);
    float3 J3 = cross(vec2, vec8) + cross(vec7, vec1);

    return Mat3f{
        J1.x, J2.x, J3.x,
        J1.y, J2.y, J3.y,
        J1.z, J2.z, J3.z
    };
}


NV_IDX_DEVICE_INLINE_MEMBER
Mat3f get_v_jacobian_numerical(const float3& sample_position,
    const SVW_Sparse_Volume_Sampler& svw_sampler, const float dx, const float dy, const float dz)
{
    const float3 ex = make_float3(1.0f, 0.0f, 0.0f);
    const float3 ey = make_float3(0.0f, 1.0f, 0.0f);
    const float3 ez = make_float3(0.0f, 0.0f, 1.0f);

    const float4 pex = svw_sampler.v_trilinear.fetch_sample(sample_position + ex);
    const float4 pey = svw_sampler.v_trilinear.fetch_sample(sample_position + ey);
    const float4 pez = svw_sampler.v_trilinear.fetch_sample(sample_position + ez);
    const float4 nex = svw_sampler.v_trilinear.fetch_sample(sample_position - ex);
    const float4 ney = svw_sampler.v_trilinear.fetch_sample(sample_position - ey);
    const float4 nez = svw_sampler.v_trilinear.fetch_sample(sample_position - ez);

    const float _02dXinv = 1.0f / ( 2.0*dx );
    const float _02dYinv = 1.0f / ( 2.0*dy );
    const float _02dZinv = 1.0f / ( 2.0*dz );

    return Mat3f{ // assemble and return the jacobian of vxw
        _02dXinv*( pex.x - nex.x ), _02dYinv*( pey.x - ney.x ), _02dZinv*( pez.x - nez.x ),
        _02dXinv*( pex.y - nex.y ), _02dYinv*( pey.y - ney.y ), _02dZinv*( pez.y - nez.y ),
        _02dXinv*( pex.z - nex.z ), _02dYinv*( pey.z - ney.z ), _02dZinv*( pez.z - nez.z )
    };
}


/*------------------------------------------------------------------------------------------------------*/
/*-------------------------------------------- root-finding --------------------------------------------*/
/*------------------------------------------------------------------------------------------------------*/


bool get_sectional_newton_descent_step(float3& step, const float3& vxw, const Mat3f& dvxw, const float __EPSILON_DET)
{
    // setup gradients
    float3 grad_dvxw_0 = make_float3(dvxw._11, dvxw._12, dvxw._13);
    float3 grad_dvxw_1 = make_float3(dvxw._21, dvxw._22, dvxw._23);
    float3 grad_dvxw_2 = make_float3(dvxw._31, dvxw._32, dvxw._33);

    // get cross products
    float3 grad_dvxw_0_cross_dvxw_1 = cross(grad_dvxw_0, grad_dvxw_1);
    float3 grad_dvxw_0_cross_dvxw_2 = cross(grad_dvxw_0, grad_dvxw_2);
    float3 grad_dvxw_1_cross_dvxw_2 = cross(grad_dvxw_1, grad_dvxw_2);

    // get magnitudes of cross products of gradients
    float grad_dvxw_0_cross_dvxw_1_squared = get_3d_norm(grad_dvxw_0_cross_dvxw_1);
    float grad_dvxw_0_cross_dvxw_2_squared = get_3d_norm(grad_dvxw_0_cross_dvxw_2);
    float grad_dvxw_1_cross_dvxw_2_squared = get_3d_norm(grad_dvxw_1_cross_dvxw_2);

    // find the two gradient vectors of dvxw that maximize the cross product magnitude
    float first_vxw = vxw.x;
    float second_vxw = vxw.y;

    float3 first_dvxw = grad_dvxw_0;
    float3 second_dvxw = grad_dvxw_1;

    float preferred_squared = grad_dvxw_0_cross_dvxw_1_squared;

    if ( preferred_squared < grad_dvxw_0_cross_dvxw_2_squared )
    {
        preferred_squared = grad_dvxw_0_cross_dvxw_2_squared;
        second_vxw = vxw.z;
        second_dvxw = grad_dvxw_2;
    }

    if ( preferred_squared < grad_dvxw_1_cross_dvxw_2_squared )
    {
        first_vxw = vxw.y;
        first_dvxw = grad_dvxw_1;
        second_vxw = vxw.z;
        second_dvxw = grad_dvxw_2;
    }

    float dot00 = dot(first_dvxw, first_dvxw);
    float dot01 = dot(first_dvxw, second_dvxw);
    float dot11 = dot(second_dvxw, second_dvxw);

    // setup linear system of equations with the found gradients
    // ( the minus sign is factorized out, see paper )

    Mat2f M = Mat2f{
        dot00, dot01,
        dot01, dot11
    };

    // M must be invertible
    float determinant = M._11*M._22 - M._12*M._21;;

    float2 b = make_float2(first_vxw, second_vxw);

    float2 st;

    if ( abs(determinant) <= __EPSILON_DET )
    {
        Mat2f M_inv = get_2d_pseudo_inverse(M);

        if ( abs(M_inv._11) <= __EPSILON_DET && abs(M_inv._22) <= __EPSILON_DET ) return false;

        st = make_float2(
            M_inv._11*b.x + M_inv._12*b.y,
            M_inv._21*b.x + M_inv._22*b.y
        );
    }
    else
    {
        st = get_2d_linear_equation_solution(M, b, determinant);
    }

    step = st.x*first_dvxw + st.y*second_dvxw;

    return true;
}


NV_IDX_DEVICE_INLINE_MEMBER
bool find_root(
    const float3& initial,
    float3& root,
    const uint N,
    const SVW_Sparse_Volume_Sampler& svw_sampler,
    const float __ADAPTIVE,
    const float __STEP_SIZE,
    const float __EPSILON_DET,
    const float __EPSILON_CHANGE,
    const float __DISTANCE_THRESHOLD,
    const float __EPSILON_INNER_ZERO,
    const float __EPSILON_OUTER_ZERO,
    const float __EPSILON_TRUST_REGION,
    const float __EPSILON_TRUST_REGION_TRIGGER)
{
    uint count = 0;
    float step_size = __STEP_SIZE;

    float3 vxw = get_vxw(root, svw_sampler);
    float residual = get_3d_norm(vxw);

    for ( count = 0; count < N; count++ )
    {
        float3 old_root = root;

        Mat3f dvxw = get_vxw_jacobian(root, svw_sampler);

        float3 step;
        bool is_det = true;

        is_det = get_sectional_newton_descent_step(step, vxw, dvxw, __EPSILON_DET);

        if ( ! is_det ) return false;

        float step_norm = get_3d_norm(step);

        // we assume convergence if not much has changed
        if ( step_norm == __EPSILON_CHANGE ) break;

        if ( step_norm >= __EPSILON_TRUST_REGION_TRIGGER )
        {
            root = root - step_size*step;
        }
        else
        {
            step = step / step_norm;
            root = root - step_size*step*min(__EPSILON_TRUST_REGION, step_norm);
        }

        // we assume the PV line is close to the initial sampling point, otherwise we abort
        if ( get_3d_norm(root - initial) > __DISTANCE_THRESHOLD ) return false;

        vxw = get_vxw(root, svw_sampler);
        float new_residual = get_3d_norm(vxw);

        // got worse? go back and decrease step

        if ( __ADAPTIVE && new_residual > residual ) {
            root = old_root;
            step_size *= 0.5;
        }
        else
        {
            residual = new_residual;
        }

        // if we're at zero, we're done
        if ( residual <= __EPSILON_INNER_ZERO ) return true;
    }

    return residual <= __EPSILON_OUTER_ZERO;
}



/*------------------------------------------------------------------------------------------------------*/
/*----------------------------------------------- shading ----------------------------------------------*/
/*------------------------------------------------------------------------------------------------------*/


NV_IDX_DEVICE_INLINE_MEMBER
Intersect get_intersect_sphere(const Ray& ray, const Sphere& sphere, const float __SPHERE_THICKNESS_TRESH)
{
    const float r = sphere.radius;
    if ( r <= 1.0e-10f ) return Intersect{false, 0.0f};
    const float3 v = ray.origin - sphere.center;
    const float b = dot(v, ray.direction);
    const float c = dot(v, v) - r*r;
    const float discriminant = b*b - c;
    if ( abs(discriminant) <= __SPHERE_THICKNESS_TRESH ) return Intersect{true, -b};
    if ( discriminant > 0.0f ) return Intersect{true, -b - sqrt(discriminant)};
    return Intersect{false, 0.0f};
}


NV_IDX_DEVICE_INLINE_MEMBER
float3 get_shading_color(const float3& shading_position, const float3& normal, const float3& color)
{
    const float diffuse_coefficient = 1.0f;
    const float3 diffuse_color = color;
    const float3 light_position = make_float3(-100.0f, -100.0f, -100.0f);

    const float3 shading_to_light = light_position - shading_position;
    const float3 light_direction = normalize(shading_to_light);

    const float ndotl = dot(normal, light_direction);
    const float diffuse_intensity = 0.5 + 0.5*ndotl;

    return diffuse_color*diffuse_intensity*diffuse_coefficient;
}


/*------------------------------------------------------------------------------------------------------*/
/*---------------------------------------------- colormap ----------------------------------------------*/
/*------------------------------------------------------------------------------------------------------*/


NV_IDX_DEVICE_INLINE_MEMBER
Eigenvalues3f get_3d_eigenvalues(const Mat3f& M)
{
    // adapted from https://github.com/tobguent/small-mcftle/blob/master/demo_gpu/flow.hlsli
    // via http://read.pudn.com/downloads21/sourcecode/graph/71499/gems/Roots3And4.c__.htm
    // Refer to https://proofwiki.org/wiki/Cardano%27s_Formula for the proof and explicit
    // formula for all solutions, including the imaginary part(s)

    const float a = M._11; const float b = M._12; const float c = M._13;
    const float d = M._21; const float e = M._22; const float f = M._23;
    const float g = M._31; const float h = M._32; const float i = M._33;

    // Ax^3 + Bx^2 + Cx + D = 0, we know that the characteristic polynomial
    // of a 3x3 matrix has A = -1.0, therefore we can multiply the equation by -1
    const float B = -( i + e + a );
    const float C = -( - e*i - a*i + f*h + c*g - a*e + b*d );
    const float D = -( a * ( e*i - f*h ) - b * ( d*i - f*g ) + c * ( d*h - e*g ) );

    const float Q = ( 3.0*C - B*B ) / ( 9.0 );
    const float R = ( 9.0*B*C - 27.0*D - 2.0*B*B*B ) / ( 54.0 );

    // discriminant
    const float DISCR = Q*Q*Q + R*R;

    const float S = cbrt( R + sqrt(DISCR) );
    const float T = cbrt( R - sqrt(DISCR) );

    Eigenvalues3f eigenvalues;

    if ( abs(DISCR) <= 1E-5 )
    {
        eigenvalues.ev1_real = S + T - B/3.0f;
        eigenvalues.ev2_real = - (S+T)/2.0f - B/3.0f;
        eigenvalues.ev3_real = - (S+T)/2.0f - B/3.0f;

        // only real eigenvalues exist
        eigenvalues.ev2_complex = 0.0f;
        eigenvalues.ev3_complex = 0.0f;
    }
    else if ( DISCR > 0.0f )
    {
        eigenvalues.ev1_real = S + T - B/3.0f;
        eigenvalues.ev2_real = - (S+T)/2.0f - B/3.0f;
        eigenvalues.ev3_real = - (S+T)/2.0f - B/3.0f;

        // complex conjugate pair of eigenvalues
        eigenvalues.ev2_complex =  (S-T)*sqrt(3.0f)/2.0f;
        eigenvalues.ev3_complex = -(S-T)*sqrt(3.0f)/2.0f;
    }
    else
    {
        // all real are unequal
        float pi = 3.141592654f;
        float theta = acos(R/sqrt(-Q*Q*Q));
        eigenvalues.ev1_real = 2.0f*sqrt(-Q)*cos(theta/3.0f) - B/3.0f;
        eigenvalues.ev2_real = 2.0f*sqrt(-Q)*cos(theta/3.0f + 2.0f*pi/3.0f) - B/3.0f;
        eigenvalues.ev3_real = 2.0f*sqrt(-Q)*cos(theta/3.0f + 4.0f*pi/3.0f) - B/3.0f;

        // only real eigenvalues exist
        eigenvalues.ev2_complex = 0.0f;
        eigenvalues.ev3_complex = 0.0f;
    }

    return eigenvalues;
}


float get_vorticity_magnitude(const float3& sample_position,
    const SVW_Sparse_Volume_Sampler& svw_sampler, const Mat3f& Jacobian)
{
    const float J32 = Jacobian._32;
    const float J23 = Jacobian._23;

    const float J13 = Jacobian._13;
    const float J31 = Jacobian._31;

    const float J21 = Jacobian._21;
    const float J12 = Jacobian._12;

    // also known as curl or rotational
    const float3 vorticity = make_float3(J32 - J23, J13 - J31, J21 - J12);

    return get_3d_norm(vorticity);
}
