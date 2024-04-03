#include <iostream>
#include "Eigen/Eigen"
#include <iomanip>

using namespace std;
using namespace Eigen;

Vector2d qrsolve(const Matrix2d& A, const Vector2d& b) //la funzione calcola la soluzione col metodo qr implementato nella libreria Eigen. Il metodo Qr di Eigen non ha richieste sulla matrice
{
    Vector2d x;
    x << A.colPivHouseholderQr().solve(b); //usa il pivoting per colonne, che è più lento ma più preciso, poiché le matrici 2x2 sono piccole e non impattano molto sulla velocità dell'algoritmo
    return x;
}

Vector2d lusolve(const Matrix2d& A, const Vector2d& b) //la funzione calcola la soluzione con la fattorizzazione PA = LU implementata in Eigen
{
    Vector2d x;
    x << A.fullPivLu().solve(b); //usa il metodo full pivoting (più lento ma più preciso rispetto al partial pivoting) poiché il partial pivoting richiede che le matrici siano invertibili e il determinante della seconda e della terza matrice è molto vicino a zero
    return x;
}

double relerr(const Vector2d& xstar, const Vector2d& x) //calcola l'errore relativo usando la funzione norm() di Eigen
{
    const double err = (x-xstar).norm()/xstar.norm();
    return err;
}

int main()
{
    Vector2d xstar;
    xstar << -1.0e+0, -1.0e+00; //soluzione reale xstar = [-1;-1]

    Matrix2d A1;
    A1 << 5.547001962252291e-01, -3.770900990025203e-02, 8.320502943378437e-01, -9.992887623566787e-01; //matrice del primo sistema
    Vector2d b1;
    b1 << -5.169911863249772e-01, 1.672384680188350e-01; //termine noto del secondo sistema
    Vector2d x1qr = qrsolve(A1,b1); //soluzione del primo sistema con qr
    cout << scientific << setprecision(16) << "L'errore relativo sul primo sistema con QR è: " << relerr(xstar,x1qr) << endl;
    Vector2d x1lu = lusolve(A1,b1); //soluzione del primo sistema con lu
    cout << scientific << setprecision(16) << "L'errore relativo sul primo sistema con LU è: " << relerr(xstar,x1lu) << endl;

    if (relerr(xstar, x1qr) < relerr(xstar,x1lu))
    {
        cout << "Per il primo sistema la fattorizzazione QR è più precisa\n" << endl;
    }
    else
    {
        cout << "Per il primo sistema la fattorizzazione LU è più precisa\n" << endl;
    }


    Matrix2d A2;
    A2 << 5.547001962252291e-01, -5.540607316466765e-01, 8.320502943378437e-01, -8.324762492991313e-01; //matrice del secondo sistema
    Vector2d b2;
    b2 << -6.394645785530173e-04, 4.259549612877223e-04; //termine noto del secondo sistema
    Vector2d x2qr = qrsolve(A2,b2); //soluzione del secondo sistema con qr
    cout << scientific << setprecision(16) << "L'errore relativo sul secondo sistema con QR è: " << relerr(xstar,x2qr) << endl;
    Vector2d x2lu = lusolve(A2,b2); //soluzione del secondo sistema con lu
    cout << scientific << setprecision(16) << "L'errore relativo sul secondo sistema con LU è: " << relerr(xstar,x2lu) << endl;

    if (relerr(xstar, x2qr) < relerr(xstar,x2lu))
    {
        cout << "Per il secondo sistema la fattorizzazione QR è più precisa\n" << endl;
    }
    else
    {
        cout << "Per il secondo sistema la fattorizzazione LU è più precisa\n" << endl;
    }


    Matrix2d A3;
    A3 << 5.547001962252291e-01, -5.547001955851905e-01, 8.320502943378437e-01, -8.320502947645361e-01; //matrice del terzo sistema
    Vector2d b3;
    b3 << -6.400391328043042e-10, 4.266924591433963e-10; //termine noto del terzo sistema
    Vector2d x3qr = qrsolve(A3,b3); //soluzione del terzo sistema con qr
    cout << scientific << setprecision(16) << "L'errore relativo sul terzo sistema con QR è: " << relerr(xstar,x3qr) << endl;
    Vector2d x3lu = lusolve(A3,b3); //soluzione del terzo sistema con lu
    cout << scientific << setprecision(16) << "L'errore relativo sul terzo sistema con LU è: " << relerr(xstar,x3lu) << endl;

    if (relerr(xstar, x3qr) < relerr(xstar,x3lu))
    {
        cout << "Per il terzo sistema la fattorizzazione QR è più precisa\n" << endl;
    }
    else
    {
        cout << "Per il terzo sistema la fattorizzazione LU è più precisa\n" << endl;
    }


    double meanlu = (relerr(xstar,x1lu) + relerr(xstar,x2lu) + relerr(xstar,x3lu))/3;
    double meanqr = (relerr(xstar,x1qr) + relerr(xstar,x2qr) + relerr(xstar,x3qr))/3;
    cout << scientific << setprecision(16) << "La media degli errori relativi delle fattorizzazioni LU è: " << meanlu << endl;
    cout << scientific << setprecision(16) << "La media degli errori relativi delle fattorizzazioni QR è: " << meanqr << endl;

    if (meanlu < meanqr)
    {
        cout << "La fattorizzazione LU è in media più precisa" << endl;
    }

    if (meanlu > meanqr)
    {
        cout << "La fattorizzazione QR è in media più precisa" << endl;
    }

    return 0;
}
