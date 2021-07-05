def odd_wide_angle_coefficients(ell, n=1):
    if n != 1:
        raise NotImplementedError('Only odd order n == 1 is supported!')

    def coefficient(ell):
        return ell*(ell+1)/(2*ell+1)

    if ell == 1:
        return [ell + 1, coefficient(ell+1)]
    return [ell-1, ell+1], [coefficient(ell-1), coefficient(ell+1)]


class PowerOddWideAngle(BaseClass):

    def __init__(self, ells, k=None, d=1., n=1):
        self.ells = ells
        self.ellsout = ells + list(range(1,max(ells)))
        self.k = k
        ells = np.array(ells)
        eye = np.eye(len(self.k),dtype=self.k.dtype)
        matrix = []
        for ellout in ellsout:
            line = []
            if ellout % 2 == 0:
                line += [(ell == ellout)*eye for ell in self.ells]
            else:
                ells,coeffs = odd_wide_angle_coefficients(ell,n=n)
                for ell in self.ells:
                    if ell in ells:
                        coeff = coeffs.index(ell)/d
                        if ell == ellout + 1:
                            coeff_spherical_bessel, coeff_spherical_bessel_derivative = ell + 1, 1.
                        else:
                            coeff_spherical_bessel, coeff_spherical_bessel_derivative = - ell, -1
                        tmp = np.diag(coeff_spherical_bessel * coeff / self.k)
                        deltak = 2. * np.diff(self.k)
                        tmp += np.diag(coeff * coeff_spherical_bessel_derivative / deltak, k=-1) - np.diag(coeff * coeff_spherical_bessel_derivative / deltak, k=1)
                        tmp[0,0] = 2.*coeff/deltak
                        tmp[0,1] = -tmp[0,0]
                        tmp[-1,-1] = -tmp[0,0]
                        tmp[-2,1] = tmp[0,0]
                    else:
                        tmp = 0.*eye
                    line.append(tmp)
            matrix.append(line)
        matrix = np.array(matrix) # now (ellsout,ells,kout,k)
        self.matrix = np.transpose(matrix,(2,0,3,1)) # now (kout,ellsout,k,ells)

    def compute(self, func):
        return np.sum(self.matrix*func,axis=(2,3))

    def __getstate__(self, state):
        for key in ['ells','ellsout','k','matrix']:
            state[key] = getattr(self,key)
        return state
