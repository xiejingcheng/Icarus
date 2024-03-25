#ifndef PI
#define PI 3.14159265358979323846
#endif

class Complex {
public:
    double real;
    double imag;

    Complex() {

    }

  
    __device__ static Complex W(int n) {
        Complex res = Complex(cos(2.0 * PI / n), sin(2.0 * PI / n));
        return res;
    }

  
    __device__ static Complex W(int n, int k) {
        Complex res = Complex(cos(2.0 * PI * k / n), sin(2.0 * PI * k / n));
        return res;
    }
    

    static Complex GetComplex(double real, double imag) {
        Complex r;
        r.real = real;
        r.imag = imag;
        return r;
    }


    static Complex GetRandomComplex() {
        Complex r;
        r.real = (double)rand() / rand();
        r.imag = (double)rand() / rand();
        return r;
    }


    static Complex GetRandomReal() {
        Complex r;
        r.real = (double)rand() / rand();
        r.imag = 0;
        return r;
    }


    static Complex GetRandomPureImag() {
        Complex r;
        r.real = 0;
        r.imag = (double)rand() / rand();
        return r;
    }


    __device__ Complex(double real, double imag) {
        this->real = real;
        this->imag = imag;
    }
    

    __device__ Complex operator+(const Complex &other) {
        Complex res(this->real + other.real, this->imag + other.imag);
        return res;
    }

    __device__ Complex operator-(const Complex &other) {
        Complex res(this->real - other.real, this->imag - other.imag);
        return res;
    }

    __device__ Complex operator*(const Complex &other) {
        Complex res(this->real * other.real - this->imag * other.imag, this->imag * other.real + this->real * other.imag);
        return res;
    }
};

