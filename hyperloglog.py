#!/usr/bin/env python3
'''
library to perform hyperloglog with 4 bit buckets

Based off of Xiao, Zhou, Chen. "Better with Fewer Bits: Improving the Performance of Cardinality Estimation of Large Data Streams". 2017, InfoCom.
'''
import numpy as np
from hashlib import sha1
from collections.abc import MutableSequence
from collections import Counter
from scipy.special import erf
from scipy.special import gammaln


class Array4Bit(MutableSequence):
    '''Array class that stores effectively uint4

    Note that we are unable to insert or remove objects, only modify them.
    '''
    def __init__(self, L):
        self.L = L
        self._array = np.zeros((L + 1) // 2, dtype=np.uint8)
        super().__init__()

    def __getitem__(self, i):
        if i >= self.L or -i > self.L:
            raise IndexError("array index out of range")
        if i < 0:
            i = self.L + i

        x = self._array[i // 2]
        if i % 2 == 0:
            y = x % 16
        else:
            y = x // 16
        return y

    def __setitem__(self, i, v):
        if i >= self.L or -i > self.L:
            raise IndexError("array assignment index out of range")
        if i < 0:
            i = self.L + i

        x = self._array[i // 2]
        y = x % 16
        z = x // 16
        if i % 2 == 0:
            y = v
        else:
            z = v
        self._array[i // 2] = y + z * 16

    def __delitem__(self, i):
        raise NotImplementedError

    def __len__(self, i):
        return self.L

    def insert(self, i):
        raise NotImplementedError

    def __repr__(self):
        return [x for x in self].__repr__()

    def to_list(self):
        return [x for x in self]

    def __eq__(self, other):
        if (self.L == other.L) and np.array_equal(self._array, other._array):
            return True
        else:
            return False


class HyperLogLog4(object):
    '''Class that stores a 4-bit HyperLogLog sketch'''
    def __init__(self, m, salt=None, items=None):
        '''m is the number of buckets, and the cryptographic salt is a prefix
        before hashing

        If salt is not set to a string, then we cannot update the HLL.

        If items is given, the sketch inserts those items.
        '''
        if salt is not None:
            if isinstance(salt, str):
                self.salt = salt
            else:
                raise TypeError("salt must be type string")
        else:
            self.salt = None
        self.m = m
        self.buckets = Array4Bit(m)
        self.base = 0  # the base register for the buckets

        if items is not None:
            if not isinstance(self.salt, str):
                raise NotImplementedError("update is only implemented if a string salt is set")
            salt = self.salt
            plist_crh = (sha1((salt + str(x)).encode()).digest() for x in items)
            plist_hash = ((int.from_bytes(temp[0:8], byteorder='big') % self.m,
                          min(64 + 1 - int.from_bytes(temp[8:16], byteorder='big').bit_length(), 63))
                          for temp in plist_crh)
            buckets_large = np.zeros(m, dtype=np.uint8)  # Because it's easier to work initially with bigger buckets
            for i, v in plist_hash:
                buckets_large[i] = max(buckets_large[i], v)
            self.base = min(buckets_large)
            for i in range(self.m):
                self.buckets[i] = min(buckets_large[i] - self.base, 15)

    def update(self, L):
        '''Inserts a list of items in L into the sketch
        '''
        update_sketch = self.__class__(self.m, salt=self.salt, items=L)
        summed = self + update_sketch
        self.buckets = summed.buckets
        self.base = summed.base

    def __eq__(self, other):
        '''Test that two HyperLogLogs are exactly the same'''
        if (self.m == other.m) and (self.buckets == other.buckets) and (self.salt == other.salt) and (self.base == other.base):
            return True
        else:
            return False

    def __add__(self, other):
        '''Combines two HyperLogLog sketches together, forming the sketch of the union'''
        assert(self.m == other.m)
        assert(self.salt == other.salt)
        ans = self.__class__(self.m, salt=self.salt)
        self_buckets_full = np.asarray(self.buckets.to_list(), dtype=np.uint8) + self.base
        other_buckets_full = np.asarray(other.buckets.to_list(), dtype=np.uint8) + other.base
        ans_buckets_full = np.maximum(self_buckets_full, other_buckets_full)
        ans.base = min(ans_buckets_full)
        for i in range(self.m):
            ans.buckets[i] = min(ans_buckets_full - ans.base, 15)
        return ans

    def __radd__(self, other):
        '''Allows summation by defining what to do if other doesn't know how to add'''
        if other == 0:
            return self
        else:
            raise TypeError('Cannot add together types')

    def count_original(self):
        '''Returns cardinality based on original HLL estimator.
        '''
        buckets = np.float64(np.asarray((self.buckets.to_list() + self.base)))
        bucketnum = len(buckets)
        if bucketnum == 16:
            alpha = 0.673
        elif bucketnum == 32:
            alpha = 0.697
        elif bucketnum == 64:
            alpha = 0.709
        else:
            alpha = 0.7213 / (1 + 1.079 / bucketnum)

        res = alpha * bucketnum**2 * 1 / sum([2**-val for val in buckets])
        if res <= (5. / 2.) * bucketnum:
            V = sum(val == 0 for val in buckets)
            if V != 0:
                res2 = bucketnum * np.log(bucketnum / V)  # linear counting
            else:
                res2 = res
        elif res <= (1. / 30.) * (1 << 32):
            res2 = res
        else:
            res2 = -(1 << 32) * np.log(1 - res / (1 << 32))
        return res2

    @classmethod
    def gmnk(cls, m, n, k):
        '''Equation 1 from https://www.dropbox.com/s/60p90gavz72spwl/infocom17-cll.pdf
            Note that here we swap the usual meaning of k and m to match the paper.
            m = number of buckets
            n = guess for count
            k = value
        '''
        mu = n / m
        sigma = np.sqrt(n / m * (1 - 1 / m))
        phi = mu + sigma**2 * np.log(1 - 2**-k)
        gamma = 2 * (1 - (1 - 1 / m)**n) / (erf(mu / (sigma * np.sqrt(2))) + erf((n - mu) / (sigma * np.sqrt(2))))
        if n <= 20 * m:
            ans = (1 - 1 / (m * 2**k))**n - (1 - 1 / m)**n
        else:
            ans = gamma * 0.5 * np.exp((phi**2 - mu**2) / (2 * sigma**2)) * (erf(phi / (sigma * np.sqrt(2))) + erf(n - phi) / (sigma * np.sqrt(2)))
        return ans

    @classmethod
    def der_gmnk(cls, m, n, k):
        '''Derivative of gmnk with respect to n.

        From (30) in https://www.dropbox.com/s/60p90gavz72spwl/infocom17-cll.pdf

        '''
        mu = n / m
        sigma = np.sqrt(n / m * (1 - 1 / m))
        phi = mu + sigma**2 * np.log(1 - 2**-k)
        dgamma_dn = (
            -((1 - 1 / m)**n * np.log(1 - 1 / m)) / (1 - (1 - 1 / m)**n)
            - 1 / (n * np.sqrt(np.pi))
            * (
                mu / (sigma * np.sqrt(2)) * np.exp(-1 * mu**2 / (2 * sigma**2))
                + (n - mu) / (sigma * np.sqrt(2)) * np.exp(-1 * (n - mu)**2 / (2 * sigma**2))
            )
            / (
                erf(mu / (sigma * np.sqrt(2))) + erf((n - mu) / (sigma * np.sqrt(2)))
            )
        )
        if n <= 20 * m:
            ans = (
                (1 - (1 / (m * 2**k)))**n * np.log(1 - 1 / (m * 2**k))
                - (1 - 1 / m)**n * np.log(1 - 1 / m)
            )
        else:
            ans = cls.gmnk(m, n, k) * (
                dgamma_dn
                + 1 / n * (phi**2 - mu**2) / (2 * sigma**2)
                + 1 / (n * np.sqrt(np.pi))
                * (
                    phi / (sigma * np.sqrt(2)) * np.exp(-1 * phi**2 / (2 * sigma**2))
                    + (n - phi) / (sigma * np.sqrt(2)) * np.exp(-1 * (n - phi)**2 / (2 * sigma**2))
                )
                / (
                    erf(phi / (sigma * np.sqrt(2))) + erf((n - phi) / (sigma * np.sqrt(2)))
                )
            )
        return ans

    @classmethod
    def pr_mjk(cls, m, n, k):
        '''Theorem 1 from https://www.dropbox.com/s/60p90gavz72spwl/infocom17-cll.pdf
            Note that here we swap the usual meaning of k and m to match the paper.
            m = number of buckets
            n = guess for count
            k = value
        '''
        if k == 0:
            return (1 - 1 / m)**n
        elif k == 1:
            return cls.gmnk(m, n, k)
        else:
            return cls.gmnk(m, n, k) - cls.gmnk(m, n, k - 1)

    @classmethod
    def der_pr_mjk(cls, m, n, k):
        '''Derivative of pr_mjk with respect to n.
        From (30) in https://www.dropbox.com/s/60p90gavz72spwl/infocom17-cll.pdf
        '''
        if k == 0:
            return (1 - 1 / m)**n * np.log(1 - 1 / m)
        elif k == 1:
            return cls.der_gmnk(m, n, k)
        else:
            return cls.der_gmnk(m, n, k) - cls.der_gmnk(m, n, k - 1)

    def log_likelihood(self, n):
        '''Returns the log likelihood of the true count being n given
        the bucket distribution we observe'''
        buckets = np.float64(np.asarray((self.buckets.to_list() + self.base)))
        bucket_freqs = Counter(buckets)
        m = len(buckets)
        ans = 0
        ans = ans + gammaln(m + 1)  # m!
        for k, nk in bucket_freqs.items():
            ans = ans + nk * np.log(self.pr_mjk(m, n, k))  # Product pr_mjk's
            if k > 0:
                ans = ans - gammaln(nk + 1)  # 1/(N1! * N2! * ... * N(K-1)!
        return ans

    def der_log_likelihood(self, n):
        '''Returns the derivative of the log-likelihood.
        From Appendix B of paper'''
        buckets = np.float64(np.asarray((self.buckets.to_list() + self.base)))
        bucket_freqs = Counter(buckets)
        m = len(buckets)
        ans = 0
        for k, nk in bucket_freqs.items():
            ans = ans + nk * self.der_pr_mjk(m, n, k) / self.pr_mjk(m, n, k)
        return ans

    def optimization_step_size(self):
        '''Computes an optimization step size based off of 2^B * m, where
        B is the smallest of the registers, and m is the number of buckets'''
        B = int(min(self.buckets))
        m = self.m
        # return (2**B * m)
        return 2**B * m

    def count_optimization(self):
        '''Returns list of guessed cardinalities based on MLE estimator optimizer'''
        raw = self.count_original()
        guess_list = [raw]
        eta = self.optimization_step_size()
        for _ in range(20):
            curr = guess_list[-1]
            guess = curr + eta * self.der_log_likelihood(curr)
            guess_list.append(guess)
        return guess_list

    def count(self):
        '''Returns cardinality based on MLE estimator'''
        return self.count_optimization()[-1]


def test_accuracy(n=10000, m=4000):
    '''Compares the accuracy of using the original vs MLE estimators

    Runs an experiment 100 random times with n items and m buckets.

    Returns the relative errors of the original HLL estimator and the MLE estimator on each experiment.
        import numpy as np
        np.mean(orig_err) = average relative bias of standard estimator
        np.mean(mle_err) = average relative bias of HLL estimator

        np.mean(np.abs(orig_err)) = average relative error of standard estimator
        np.mean(np.abs(mle_err)) = average relative error of MLE estimator

        np.mean(np.abs(final_delta)) = measurement of convergence. How much the MLE estimator changes on the 20th step (relatively).
    '''
    orig_err = []
    mle_err = []
    final_delta = []
    for _ in range(100):
        guess_list = HyperLogLog4(m, salt=str(np.random.randint(100000)), items=np.arange(n)).count_optimization()
        orig_err.append((guess_list[0] - n) / n)
        mle_err.append((guess_list[-1] - n) / n)
        final_delta.append((guess_list[-1] - guess_list[-2]) / n)
    return (orig_err, mle_err, final_delta)
