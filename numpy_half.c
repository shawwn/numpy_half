/*
 * IEEE Half-Precision Floating Point Type for NumPy
 * Copyright (c) 2010, Mark Wiebe
 *
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NumPy Developers nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTERS BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/npy_math.h>

#define NPY_PY3K 1

#include "bfloat16.h"





#if 0

// Python type for PyBfloat16 objects.
PyTypeObject bfloat16_type = {
    PyVarObject_HEAD_INIT(nullptr, 0) "bfloat16",  // tp_name
    sizeof(PyBfloat16),                            // tp_basicsize
    0,                                             // tp_itemsize
    nullptr,                                       // tp_dealloc
#if PY_VERSION_HEX < 0x03080000
    nullptr,  // tp_print
#else
    0,  // tp_vectorcall_offset
#endif
    nullptr,               // tp_getattr
    nullptr,               // tp_setattr
    nullptr,               // tp_compare / tp_reserved
    PyBfloat16_Repr,       // tp_repr
    &PyBfloat16_AsNumber,  // tp_as_number
    nullptr,               // tp_as_sequence
    nullptr,               // tp_as_mapping
    PyBfloat16_Hash,       // tp_hash
    nullptr,               // tp_call
    PyBfloat16_Str,        // tp_str
    nullptr,               // tp_getattro
    nullptr,               // tp_setattro
    nullptr,               // tp_as_buffer
                           // tp_flags
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    "bfloat16 floating-point values",  // tp_doc
    nullptr,                           // tp_traverse
    nullptr,                           // tp_clear
    PyBfloat16_RichCompare,            // tp_richcompare
    0,                                 // tp_weaklistoffset
    nullptr,                           // tp_iter
    nullptr,                           // tp_iternext
    nullptr,                           // tp_methods
    nullptr,                           // tp_members
    nullptr,                           // tp_getset
    nullptr,                           // tp_base
    nullptr,                           // tp_dict
    nullptr,                           // tp_descr_get
    nullptr,                           // tp_descr_set
    0,                                 // tp_dictoffset
    nullptr,                           // tp_init
    nullptr,                           // tp_alloc
    PyBfloat16_New,                    // tp_new
    nullptr,                           // tp_free
    nullptr,                           // tp_is_gc
    nullptr,                           // tp_bases
    nullptr,                           // tp_mro
    nullptr,                           // tp_cache
    nullptr,                           // tp_subclasses
    nullptr,                           // tp_weaklist
    nullptr,                           // tp_del
    0,                                 // tp_version_tag
};

// Numpy support

PyArray_ArrFuncs NPyBfloat16_ArrFuncs;

PyArray_Descr NPyBfloat16_Descr = {
    PyObject_HEAD_INIT(nullptr)  //
                                 /*typeobj=*/
    (&bfloat16_type),
    // We must register bfloat16 with a kind other than "f", because numpy
    // considers two types with the same kind and size to be equal, but
    // float16 != bfloat16.
    // The downside of this is that NumPy scalar promotion does not work with
    // bfloat16 values.
    /*kind=*/'V',
    // TODO(phawkins): there doesn't seem to be a way of guaranteeing a type
    // character is unique.
    /*type=*/'E',
    /*byteorder=*/'=',
    /*flags=*/NPY_NEEDS_PYAPI | NPY_USE_GETITEM | NPY_USE_SETITEM,
    /*type_num=*/0,
    /*elsize=*/sizeof(bfloat16),
    /*alignment=*/alignof(bfloat16),
    /*subarray=*/nullptr,
    /*fields=*/nullptr,
    /*names=*/nullptr,
    /*f=*/&NPyBfloat16_ArrFuncs,
    /*metadata=*/nullptr,
    /*c_metadata=*/nullptr,
    /*hash=*/-1,  // -1 means "not computed yet".
};
#endif






typedef struct {
        PyObject_HEAD
        npy_half obval;
} PyXHalfScalarObject;


PyTypeObject PyXHalfArrType_Type = {
#if defined(NPY_PY3K)
    PyVarObject_HEAD_INIT(NULL, 0)
#else
    PyObject_HEAD_INIT(NULL)
    0,                                          /* ob_size */
#endif
    "half.bfloat16",                            /* tp_name*/
    sizeof(PyXHalfScalarObject),                 /* tp_basicsize*/
    0,                                          /* tp_itemsize */
    0,                                          /* tp_dealloc */
    0,                                          /* tp_print */
    0,                                          /* tp_getattr */
    0,                                          /* tp_setattr */
#if defined(NPY_PY3K)
    0,                                          /* tp_reserved */
#else
    0,                                          /* tp_compare */
#endif
    0,                                          /* tp_repr */
    0,                                          /* tp_as_number */
    0,                                          /* tp_as_sequence */
    0,                                          /* tp_as_mapping */
    0,                                          /* tp_hash */
    0,                                          /* tp_call */
    0,                                          /* tp_str */
    0,                                          /* tp_getattro */
    0,                                          /* tp_setattro */
    0,                                          /* tp_as_buffer */
    0,                                          /* tp_flags */
    0,                                          /* tp_doc */
    0,                                          /* tp_traverse */
    0,                                          /* tp_clear */
    0,                                          /* tp_richcompare */
    0,                                          /* tp_weaklistoffset */
    0,                                          /* tp_iter */
    0,                                          /* tp_iternext */
    0,                                          /* tp_methods */
    0,                                          /* tp_members */
    0,                                          /* tp_getset */
    0,                                          /* tp_base */
    0,                                          /* tp_dict */
    0,                                          /* tp_descr_get */
    0,                                          /* tp_descr_set */
    0,                                          /* tp_dictoffset */
    0,                                          /* tp_init */
    0,                                          /* tp_alloc */
    0,                                          /* tp_new */
    0,                                          /* tp_free */
    0,                                          /* tp_is_gc */
    0,                                          /* tp_bases */
    0,                                          /* tp_mro */
    0,                                          /* tp_cache */
    0,                                          /* tp_subclasses */
    0,                                          /* tp_weaklist */
    0,                                          /* tp_del */
#if PY_VERSION_HEX >= 0x02060000
    0,                                          /* tp_version_tag */
#endif
};

static PyArray_ArrFuncs _PyXHalf_ArrFuncs;
PyArray_Descr *half_descr;


PyArray_Descr xfloat16_Descr = {
    PyObject_HEAD_INIT(nullptr)  //
                                 /*typeobj=*/
    (&PyXHalfArrType_Type),
    // We must register bfloat16 with a kind other than "f", because numpy
    // considers two types with the same kind and size to be equal, but
    // float16 != bfloat16.
    // The downside of this is that NumPy scalar promotion does not work with
    // bfloat16 values.
    /*kind=*/'V',
    // TODO(phawkins): there doesn't seem to be a way of guaranteeing a type
    // character is unique.
    /*type=*/'E',
    /*byteorder=*/'=',
    /*flags=*/NPY_NEEDS_PYAPI | NPY_USE_GETITEM | NPY_USE_SETITEM,
    /*type_num=*/0,
    /*elsize=*/sizeof(npy_half),
    /*alignment=*/sizeof(npy_half), /*alignof(npy_half),*/
    /*subarray=*/nullptr,
    /*fields=*/nullptr,
    /*names=*/nullptr,
    /*f=*/&_PyXHalf_ArrFuncs,
    /*metadata=*/nullptr,
    /*c_metadata=*/nullptr,
    /*hash=*/-1,  // -1 means "not computed yet".
};

#if 0
    half_descr = PyObject_New(PyArray_Descr, &PyArrayDescr_Type);
    half_descr->typeobj = &PyXHalfArrType_Type;
    half_descr->kind = 'f';
    half_descr->type = 'j';
    half_descr->byteorder = '=';
    half_descr->type_num = 0; /* assigned at registration */
    half_descr->elsize = 2;
    half_descr->alignment = 2;
    half_descr->subarray = NULL;
    half_descr->fields = NULL;
    half_descr->names = NULL;
    half_descr->f = &_PyXHalf_ArrFuncs;
#endif





static npy_half
MyPyFloat_AsHalf(PyObject *obj)
{
    double d;
    PyObject *num;

    if (obj == Py_None) {
        d = NPY_NAN;
    } else {
        num = PyNumber_Float(obj);
        if (num == NULL) {
            d = NPY_NAN;
        } else {
            d = PyFloat_AsDouble(num);
            Py_DECREF(num);
        }
    }
    return double_to_half(d);
}

static PyObject *
HALF_getitem(char *ip, PyArrayObject *ap)
{
    npy_half t1;
    double t2;

    if ((ap == NULL) || PyArray_ISBEHAVED_RO(ap)) {
        t1 = *((npy_half *)ip);
    }
    else {
        ap->descr->f->copyswap(&t1, ip, !PyArray_ISNOTSWAPPED(ap), ap);
    }
    t2 = half_to_double(t1);
    return PyFloat_FromDouble(t2);
}

static int HALF_setitem(PyObject *op, char *ov, PyArrayObject *ap)
{
    npy_half temp; /* ensures alignment */

    if (PyArray_IsScalar(op, Half)) {
        temp = ((PyXHalfScalarObject *)op)->obval;
    }
    else {
        temp = MyPyFloat_AsHalf(op);
    }
    if (PyErr_Occurred()) {
        if (PySequence_Check(op)) {
            PyErr_Clear();
            PyErr_SetString(PyExc_ValueError,
                    "setting an array element with a sequence.");
        }
        return -1;
    }
    if (ap == NULL || PyArray_ISBEHAVED(ap))
        *((npy_half *)ov)=temp;
    else {
        ap->descr->f->copyswap(ov, &temp, !PyArray_ISNOTSWAPPED(ap), ap);
    }
    return 0;

}

static int
HALF_compare (npy_half *pa, npy_half *pb, PyArrayObject *NPY_UNUSED(ap))
{
    npy_half a = *pa, b = *pb;
    npy_bool anan, bnan;
    int ret;

    anan = half_isnan(a);
    bnan = half_isnan(b);

    if (anan) {
        ret = bnan ? 0 : -1;
    } else if (bnan) {
        ret = 1;
    } else if(half_lt_nonan(a, b)) {
        ret = -1;
    } else if(half_lt_nonan(b, a)) {
        ret = 1;
    } else {
        ret = 0;
    }

    return ret;
}

static int
HALF_argmax(npy_half *ip, npy_intp n, npy_intp *max_ind, PyArrayObject *NPY_UNUSED(aip))
{
    npy_intp i;
    npy_half mp = *ip;

    *max_ind = 0;

    if (half_isnan(mp)) {
        /* nan encountered; it's maximal */
        return 0;
    }

    for (i = 1; i < n; i++) {
        ip++;
        /*
         * Propagate nans, similarly as max() and min()
         */
        if (!(half_le(*ip, mp))) {  /* negated, for correct nan handling */
            mp = *ip;
            *max_ind = i;
            if (half_isnan(mp)) {
                /* nan encountered, it's maximal */
                break;
            }
        }
    }
    return 0;
}

static void
HALF_dot(char *ip1, npy_intp is1, char *ip2, npy_intp is2, char *op, npy_intp n,
           void *NPY_UNUSED(ignore))
{
    float tmp = 0.0f;
    npy_intp i;

    for (i = 0; i < n; i++, ip1 += is1, ip2 += is2) {
        tmp += half_to_float(*((npy_half *)ip1)) *
               half_to_float(*((npy_half *)ip2));
    }
    *((npy_half *)op) = float_to_half(tmp);
}

/*
 * NumPyOS isn't exported, so skipping scan and fromstr
static int
HALF_scan(FILE *fp, npy_half *ip, void *NPY_UNUSED(ignore), PyArray_Descr *NPY_UNUSED(ignored))
{
    double result;
    int ret;

    ret = NumPyOS_ascii_ftolf(fp, &result);
    *ip = double_to_half(result);
    return ret;
}

static int
HALF_fromstr(char *str, npy_half *ip, char **endptr, PyArray_Descr *NPY_UNUSED(ignore))
{
    double result;

    result = NumPyOS_ascii_strtod(str, endptr);
    *ip = double_to_half(result);
    return 0;
}
*/

static npy_bool
HALF_nonzero (char *ip, PyArrayObject *ap)
{
    if (ap == NULL || PyArray_ISBEHAVED_RO(ap)) {
        npy_half *ptmp = (npy_half *)ip;
        return (npy_bool) (((*ptmp)&0x7fff) != 0);
    }
    else {
        npy_half tmp;
        ap->descr->f->copyswap(&tmp, ip, !PyArray_ISNOTSWAPPED(ap), ap);
        return (npy_bool) ((tmp&0x7fff) != 0);
    }
}

static void
HALF_fill(npy_half *buffer, npy_intp length, void *NPY_UNUSED(ignored))
{
    npy_intp i;
    float start = half_to_float(buffer[0]);
    float delta = half_to_float(buffer[1]);

    delta -= start;
    for (i = 2; i < length; ++i) {
        buffer[i] = float_to_half(start + i*delta);
    }
}

static void
HALF_fillwithscalar(npy_half *buffer, npy_intp length, npy_half *value, void *NPY_UNUSED(ignored))
{
    npy_intp i;
    npy_half val = *value;

    for (i = 0; i < length; ++i) {
        buffer[i] = val;
    }
}

static void
HALF_to_FLOAT(npy_half *ip, npy_uint32 *op, npy_intp n,
               PyArrayObject *NPY_UNUSED(aip), PyArrayObject *NPY_UNUSED(aop))
{
    while (n--) {
        *op++ = halfbits_to_floatbits(*ip++);
    }
}

static void
HALF_to_DOUBLE(npy_half *ip, npy_uint64 *op, npy_intp n,
               PyArrayObject *NPY_UNUSED(aip), PyArrayObject *NPY_UNUSED(aop))
{
    while (n--) {
        *op++ = halfbits_to_doublebits(*ip++);
    }
}

static void
HALF_to_LONGDOUBLE(npy_half *ip, npy_longdouble *op, npy_intp n,
               PyArrayObject *NPY_UNUSED(aip), PyArrayObject *NPY_UNUSED(aop))
{
    npy_uint32 temp;

    while (n--) {
        temp = halfbits_to_floatbits(*ip++);
        *op++ = *((double*)&temp);
    }
}

static void
HALF_to_CFLOAT(npy_half *ip, npy_uint32 *op, npy_intp n,
               PyArrayObject *NPY_UNUSED(aip), PyArrayObject *NPY_UNUSED(aop))
{
    while (n--) {
        *op++ = halfbits_to_floatbits(*ip++);
        *op++ = 0;
    }
}

static void
HALF_to_CDOUBLE(npy_half *ip, npy_uint64 *op, npy_intp n,
               PyArrayObject *NPY_UNUSED(aip), PyArrayObject *NPY_UNUSED(aop))
{
    while (n--) {
        *op++ = halfbits_to_doublebits(*ip++);
        *op++ = 0;
    }
}

static void
HALF_to_CLONGDOUBLE(npy_half *ip, npy_longdouble *op, npy_intp n,
               PyArrayObject *NPY_UNUSED(aip), PyArrayObject *NPY_UNUSED(aop))
{
    npy_uint32 temp;

    while (n--) {
        temp = halfbits_to_floatbits(*ip++);
        *op++ = *((float*)&temp);
        *op++ = 0.0;
    }
}

#define MAKE_HALF_TO_T(TYPE, type)                                             \
static void                                                                    \
HALF_to_ ## TYPE(npy_half *ip, type *op, npy_intp n,                           \
               PyArrayObject *NPY_UNUSED(aip), PyArrayObject *NPY_UNUSED(aop)) \
{                                                                              \
    npy_uint32 temp;                                                           \
                                                                               \
    while (n--) {                                                              \
        temp = halfbits_to_floatbits(*ip++);                                   \
        *op++ = (type)*((float*)&temp);                                        \
    }                                                                          \
}

MAKE_HALF_TO_T(BOOL, npy_bool);
MAKE_HALF_TO_T(BYTE, npy_byte);
MAKE_HALF_TO_T(UBYTE, npy_ubyte);
MAKE_HALF_TO_T(SHORT, npy_short);
MAKE_HALF_TO_T(USHORT, npy_ushort);
MAKE_HALF_TO_T(INT, npy_int);
MAKE_HALF_TO_T(UINT, npy_uint);
MAKE_HALF_TO_T(LONG, npy_long);
MAKE_HALF_TO_T(ULONG, npy_ulong);
MAKE_HALF_TO_T(LONGLONG, npy_longlong);
MAKE_HALF_TO_T(ULONGLONG, npy_ulonglong);
 
static void
FLOAT_to_HALF(npy_uint32 *ip, npy_half *op, npy_intp n,
               PyArrayObject *NPY_UNUSED(aip), PyArrayObject *NPY_UNUSED(aop))
{
    while (n--) {
        *op++ = floatbits_to_halfbits(*ip++);
    }
}
 
static void
DOUBLE_to_HALF(npy_uint64 *ip, npy_half *op, npy_intp n,
               PyArrayObject *NPY_UNUSED(aip), PyArrayObject *NPY_UNUSED(aop))
{
    while (n--) {
        *op++ = doublebits_to_halfbits(*ip++);
    }
}

static void
LONGDOUBLE_to_HALF(npy_longdouble *ip, npy_half *op, npy_intp n,
               PyArrayObject *NPY_UNUSED(aip), PyArrayObject *NPY_UNUSED(aop))
{
    npy_uint64 temp;

    while (n--) {
        *((double*)&temp) = (double)(*ip++);
        *op++ = doublebits_to_halfbits(temp);
    }
}

static void
CFLOAT_to_HALF(npy_uint32 *ip, npy_half *op, npy_intp n,
               PyArrayObject *NPY_UNUSED(aip), PyArrayObject *NPY_UNUSED(aop))
{
    while (n--) {
        *op++ = floatbits_to_halfbits(*ip);
        ip += 2;
    }
}

static void
CDOUBLE_to_HALF(npy_uint64 *ip, npy_half *op, npy_intp n,
               PyArrayObject *NPY_UNUSED(aip), PyArrayObject *NPY_UNUSED(aop))
{
    while (n--) {
        *op++ = doublebits_to_halfbits(*ip);
        ip += 2;
    }
}

static void
CLONGDOUBLE_to_HALF(npy_longdouble *ip, npy_half *op, npy_intp n,
               PyArrayObject *NPY_UNUSED(aip), PyArrayObject *NPY_UNUSED(aop))
{
    npy_uint64 temp;

    while (n--) {
        *((double*)&temp) = (double)(*ip);
        *op++ = doublebits_to_halfbits(temp);
        ip += 2;
    }
}


#define MAKE_T_TO_HALF(TYPE, type)                                             \
static void                                                                    \
TYPE ## _to_HALF(type *ip, npy_half *op, npy_intp n,                           \
               PyArrayObject *NPY_UNUSED(aip), PyArrayObject *NPY_UNUSED(aop)) \
{                                                                              \
    float temp;                                                                \
                                                                               \
    while (n--) {                                                              \
        temp = (float)(*ip++);                                                 \
        *op++ = float_to_half(temp);                                          \
    }                                                                          \
}

MAKE_T_TO_HALF(BOOL, npy_bool);
MAKE_T_TO_HALF(BYTE, npy_byte);
MAKE_T_TO_HALF(UBYTE, npy_ubyte);
MAKE_T_TO_HALF(SHORT, npy_short);
MAKE_T_TO_HALF(USHORT, npy_ushort);
MAKE_T_TO_HALF(INT, npy_int);
MAKE_T_TO_HALF(UINT, npy_uint);
MAKE_T_TO_HALF(LONG, npy_long);
MAKE_T_TO_HALF(ULONG, npy_ulong);
MAKE_T_TO_HALF(LONGLONG, npy_longlong);
MAKE_T_TO_HALF(ULONGLONG, npy_ulonglong);


static void register_cast_function(int sourceType, int destType, PyArray_VectorUnaryFunc *castfunc)
{
    PyArray_Descr *descr = PyArray_DescrFromType(sourceType);
    PyArray_RegisterCastFunc(descr, destType, castfunc);
    Py_DECREF(descr);
}

static PyObject *
half_arrtype_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    PyObject *obj = NULL;
    npy_half value = 0;

    if (!PyArg_ParseTuple(args, "|O", &obj)) {
        return NULL;
    }
    if (obj != NULL) {
        value = MyPyFloat_AsHalf(obj);
    }
    return PyArray_Scalar(&value, &xfloat16_Descr, NULL);
}

static PyObject *
gentype_richcompare(PyObject *self, PyObject *other, int cmp_op)
{
    PyObject *arr, *ret;

    arr = PyArray_FromScalar(self, NULL);
    if (arr == NULL) {
        return NULL;
    }
    ret = Py_TYPE(arr)->tp_richcompare(arr, other, cmp_op);
    Py_DECREF(arr);
    return ret;
}

static long
halftype_hash(PyObject *obj)
{
    double temp;
    temp = half_to_double(((PyXHalfScalarObject *)obj)->obval);
    return _Py_HashDouble(*((double*)&temp));
}

PyObject* halftype_repr(PyObject *o)
{
    double temp;
    char str[48];

    temp = half_to_double(((PyXHalfScalarObject *)o)->obval);
    sprintf(str, "bfloat16(%g)", *((double*)&temp));
    return PyUnicode_FromString(str);
}

PyObject* halftype_str(PyObject *o)
{
    double temp;
    char str[48];

    temp = half_to_double(((PyXHalfScalarObject *)o)->obval);
    sprintf(str, "%g", *((double*)&temp));
    return PyUnicode_FromString(str);
}

#if PY_MAJOR_VERSION >= 3
  #define MOD_ERROR_VAL NULL
  #define MOD_SUCCESS_VAL(val) val
  #define MOD_INIT(name) PyMODINIT_FUNC PyInit_##name(void)
  #define MOD_DEF(ob, name, doc, methods) \
          static struct PyModuleDef moduledef = { \
            PyModuleDef_HEAD_INIT, name, doc, -1, methods, }; \
          ob = PyModule_Create(&moduledef);
#else
  #define MOD_ERROR_VAL
  #define MOD_SUCCESS_VAL(val)
  #define MOD_INIT(name) void init##name(void)
  #define MOD_DEF(ob, name, doc, methods) \
          ob = Py_InitModule3(name, methods, doc);
#endif


static PyMethodDef HalfMethods[] = {
    {NULL, NULL, 0, NULL}
};
const char* module___doc__ = "";

PyMODINIT_FUNC PyInit_numpy_xhalf(void)
{
    PyObject *m;
    int halfNum;
    PyArray_Descr *descr;

    m = NULL;

    //m = Py_InitModule("numpy_xhalf", HalfMethods);
#if 1
    MOD_DEF(m, "numpy_xhalf", module___doc__,
            HalfMethods)
#endif
    if (m == NULL) {
        return NULL;
    }

    /* Make sure NumPy is initialized */
    import_array();

    /* Register the half array scalar type */
#if defined(NPY_PY3K)
    PyXHalfArrType_Type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
#else
    PyXHalfArrType_Type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_CHECKTYPES;
#endif
    PyXHalfArrType_Type.tp_new = half_arrtype_new;
    PyXHalfArrType_Type.tp_richcompare = gentype_richcompare;
    PyXHalfArrType_Type.tp_hash = halftype_hash;
    PyXHalfArrType_Type.tp_repr = halftype_repr;
    PyXHalfArrType_Type.tp_str = halftype_str;
    /* PyXHalfArrType_Type.tp_base = &PyFloatingArrType_Type; */
    PyXHalfArrType_Type.tp_base = &PyFloatingArrType_Type;
    if (PyType_Ready(&PyXHalfArrType_Type) < 0) {
        PyErr_Print();
        PyErr_SetString(PyExc_SystemError, "could not initialize PyXHalfArrType_Type");
        return NULL;
    }

    /* The array functions */
    PyArray_InitArrFuncs(&_PyXHalf_ArrFuncs);
    _PyXHalf_ArrFuncs.getitem = (PyArray_GetItemFunc*)HALF_getitem;
    _PyXHalf_ArrFuncs.setitem = (PyArray_SetItemFunc*)HALF_setitem;
    /* copy copyswap and copyswapn from uint16 */
    descr = PyArray_DescrFromType(NPY_UINT16);
    _PyXHalf_ArrFuncs.copyswap = descr->f->copyswap;
    _PyXHalf_ArrFuncs.copyswapn = descr->f->copyswapn;
    Py_DECREF(descr);
    _PyXHalf_ArrFuncs.compare = (PyArray_CompareFunc*)HALF_compare;
    _PyXHalf_ArrFuncs.argmax = (PyArray_ArgFunc*)HALF_argmax;
    _PyXHalf_ArrFuncs.dotfunc = (PyArray_DotFunc*)HALF_dot;
    /*
    _PyXHalf_ArrFuncs.scanfunc = (PyArray_ScanFunc*)HALF_scan;
    _PyXHalf_ArrFuncs.fromstr = (PyArray_FromStrFunc*)HALF_fromstr;
    */
    _PyXHalf_ArrFuncs.nonzero = (PyArray_NonzeroFunc*)HALF_nonzero;
    _PyXHalf_ArrFuncs.fill = (PyArray_FillFunc*)HALF_fill;
    _PyXHalf_ArrFuncs.fillwithscalar = (PyArray_FillWithScalarFunc*)HALF_fillwithscalar;
    _PyXHalf_ArrFuncs.cast[NPY_BOOL] = (PyArray_VectorUnaryFunc*)HALF_to_BOOL;
    _PyXHalf_ArrFuncs.cast[NPY_BYTE] = (PyArray_VectorUnaryFunc*)HALF_to_BYTE;
    _PyXHalf_ArrFuncs.cast[NPY_UBYTE] = (PyArray_VectorUnaryFunc*)HALF_to_UBYTE;
    _PyXHalf_ArrFuncs.cast[NPY_SHORT] = (PyArray_VectorUnaryFunc*)HALF_to_SHORT;
    _PyXHalf_ArrFuncs.cast[NPY_USHORT] = (PyArray_VectorUnaryFunc*)HALF_to_USHORT;
    _PyXHalf_ArrFuncs.cast[NPY_INT] = (PyArray_VectorUnaryFunc*)HALF_to_INT;
    _PyXHalf_ArrFuncs.cast[NPY_UINT] = (PyArray_VectorUnaryFunc*)HALF_to_UINT;
    _PyXHalf_ArrFuncs.cast[NPY_LONG] = (PyArray_VectorUnaryFunc*)HALF_to_LONG;
    _PyXHalf_ArrFuncs.cast[NPY_ULONG] = (PyArray_VectorUnaryFunc*)HALF_to_ULONG;
    _PyXHalf_ArrFuncs.cast[NPY_LONGLONG] = (PyArray_VectorUnaryFunc*)HALF_to_LONGLONG;
    _PyXHalf_ArrFuncs.cast[NPY_ULONGLONG] = (PyArray_VectorUnaryFunc*)HALF_to_ULONGLONG;
    _PyXHalf_ArrFuncs.cast[NPY_FLOAT] = (PyArray_VectorUnaryFunc*)HALF_to_FLOAT;
    _PyXHalf_ArrFuncs.cast[NPY_DOUBLE] = (PyArray_VectorUnaryFunc*)HALF_to_DOUBLE;
    _PyXHalf_ArrFuncs.cast[NPY_LONGDOUBLE] = (PyArray_VectorUnaryFunc*)HALF_to_LONGDOUBLE;
    _PyXHalf_ArrFuncs.cast[NPY_CFLOAT] = (PyArray_VectorUnaryFunc*)HALF_to_CFLOAT;
    _PyXHalf_ArrFuncs.cast[NPY_CDOUBLE] = (PyArray_VectorUnaryFunc*)HALF_to_CDOUBLE;
    _PyXHalf_ArrFuncs.cast[NPY_CLONGDOUBLE] = (PyArray_VectorUnaryFunc*)HALF_to_CLONGDOUBLE;

#if 0
    /* The half array descr */
    half_descr = PyObject_New(PyArray_Descr, &PyArrayDescr_Type);
    half_descr->typeobj = &PyXHalfArrType_Type;
    half_descr->kind = 'f';
    half_descr->type = 'j';
    half_descr->byteorder = '=';
    half_descr->type_num = 0; /* assigned at registration */
    half_descr->elsize = 2;
    half_descr->alignment = 2;
    half_descr->subarray = NULL;
    half_descr->fields = NULL;
    half_descr->names = NULL;
    half_descr->f = &_PyXHalf_ArrFuncs;
#endif


    Py_INCREF(&PyXHalfArrType_Type);
    Py_TYPE(&xfloat16_Descr) = &PyArrayDescr_Type;
    halfNum = PyArray_RegisterDataType(&xfloat16_Descr);

    if (halfNum < 0)
        return NULL;

    register_cast_function(NPY_BOOL, halfNum, (PyArray_VectorUnaryFunc*)BOOL_to_HALF);
    register_cast_function(NPY_BYTE, halfNum, (PyArray_VectorUnaryFunc*)BYTE_to_HALF);
    register_cast_function(NPY_UBYTE, halfNum, (PyArray_VectorUnaryFunc*)UBYTE_to_HALF);
    register_cast_function(NPY_SHORT, halfNum, (PyArray_VectorUnaryFunc*)SHORT_to_HALF);
    register_cast_function(NPY_USHORT, halfNum, (PyArray_VectorUnaryFunc*)USHORT_to_HALF);
    register_cast_function(NPY_INT, halfNum, (PyArray_VectorUnaryFunc*)INT_to_HALF);
    register_cast_function(NPY_UINT, halfNum, (PyArray_VectorUnaryFunc*)UINT_to_HALF);
    register_cast_function(NPY_LONG, halfNum, (PyArray_VectorUnaryFunc*)LONG_to_HALF);
    register_cast_function(NPY_ULONG, halfNum, (PyArray_VectorUnaryFunc*)ULONG_to_HALF);
    register_cast_function(NPY_LONGLONG, halfNum, (PyArray_VectorUnaryFunc*)LONGLONG_to_HALF);
    register_cast_function(NPY_ULONGLONG, halfNum, (PyArray_VectorUnaryFunc*)ULONGLONG_to_HALF);
    register_cast_function(NPY_FLOAT, halfNum, (PyArray_VectorUnaryFunc*)FLOAT_to_HALF);
    register_cast_function(NPY_DOUBLE, halfNum, (PyArray_VectorUnaryFunc*)DOUBLE_to_HALF);
    register_cast_function(NPY_LONGDOUBLE, halfNum, (PyArray_VectorUnaryFunc*)LONGDOUBLE_to_HALF);
    register_cast_function(NPY_CFLOAT, halfNum, (PyArray_VectorUnaryFunc*)CFLOAT_to_HALF);
    register_cast_function(NPY_CDOUBLE, halfNum, (PyArray_VectorUnaryFunc*)CDOUBLE_to_HALF);
    register_cast_function(NPY_CLONGDOUBLE, halfNum, (PyArray_VectorUnaryFunc*)CLONGDOUBLE_to_HALF);

    PyArray_RegisterCanCast(&xfloat16_Descr, NPY_FLOAT, NPY_NOSCALAR);
    PyArray_RegisterCanCast(&xfloat16_Descr, NPY_DOUBLE, NPY_NOSCALAR);
    PyArray_RegisterCanCast(&xfloat16_Descr, NPY_LONGDOUBLE, NPY_NOSCALAR);
    PyArray_RegisterCanCast(&xfloat16_Descr, NPY_CFLOAT, NPY_NOSCALAR);
    PyArray_RegisterCanCast(&xfloat16_Descr, NPY_CDOUBLE, NPY_NOSCALAR);
    PyArray_RegisterCanCast(&xfloat16_Descr, NPY_CLONGDOUBLE, NPY_NOSCALAR);

    PyModule_AddObject(m, "bfloat16", (PyObject *)&PyXHalfArrType_Type);
    return m;
}
