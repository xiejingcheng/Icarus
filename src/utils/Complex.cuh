class Complex{
public:
    double real;
    double imag;

    Complex(){

    }

    //构造函数
    __device__ Complex(double real , double imag){
        Complex c;
        this->real = real;
        this->imag = imag;
        return c;
    }

    //获取单位根
    __device__ static Complex w(int n){
        Complex res = Complex(cos(2.0 * PI / n), sin(2.0 * PI / n));
        return res;
    }

    __device__ static Complex w(int n , int k){
        Complex res = Complex(cos(2.0 * PI * k / n), sin(2.0 * PI * k / n));
        return res;
    }

    __device__ Complex operator+(const Complex &other){
        Complex res = Complex(this->real + other.real , this->imag + other.imag);
        return res;
    }

    __device__ Complex operator-(const Complex &other){
        Complex res = Complex(this->real - other.real , this->imag - other.imag);
        return res;
    }

    __device__ Complex operator*(const Complex &other){
        Complex res = Complex(this->real * other.real - this->imag * other.imag , this->imag * other.real + this->real * other.imag);
        return res;
    }
}