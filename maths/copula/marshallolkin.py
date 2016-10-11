import numpy as np
from scipy import special
from scipy.stats import expon
from scipy.optimize import brentq
from credit.default_models import StepwiseConstantIntensity


class MarshallOlkinCopula(object):
    def __init__(self, reduced_index, total_number, indices, lambdas):
        assert reduced_index >= 0, "The reduced_index must be positive"
        self.__reduced_index = reduced_index

        assert(reduced_index <= total_number), "The number of indexes is leq than the reduced_index"
        self.__dimension = total_number

        if indices and lambdas:
            self.__subsets = np.array(indices)
            self.__lambdas = np.array(lambdas)
        else:
            raise NotImplementedError("Please give subsets of indexes and lambdas.")

        self.__gamma = self.compute_gamma(self.__reduced_index)
        self.__surv_subsets_ind = self.get_remaining_indexes_in_reduced_model(self.__reduced_index)

    def compute_gamma(self, reduced_index):
        gamma = 0
        for i in np.arange(self.__subsets.size):
            s = self.__subsets[i]
            if reduced_index in s:
                gamma += self.__lambdas[i]

        return gamma

    def get_remaining_indexes_in_reduced_model(self, reduced_index):
        res = []
        for i in np.arange(self.__subsets.size):
            s = self.__subsets[i]
            if reduced_index not in s:
                res.append(i)

        return np.array(res)

    def simulate_default_times(self, number=1, use_init_indexes=True, not_included_index=None):
        lambdas = self.__lambdas
        if use_init_indexes:
            lambdas = self.defaultable_intensities
        else:
            if not_included_index:
                cp_subsets = self.get_counterparties_indices(not_included_index)
                lambdas = self.__lambdas[cp_subsets]
            else:
                raise ValueError("The not_included_index has not been given")

        U = np.random.uniform(size=(number, len(lambdas)))
        rvs = -np.log(U) / lambdas
        return rvs

    @property
    def defaultable_subsets(self):
        return np.array(self.__subsets[self.__surv_subsets_ind], copy=True)

    @property
    def defaultable_intensities(self):
        return np.array(self.__lambdas[self.__surv_subsets_ind], copy=True)

    @classmethod
    def generate_subsets_and_intensities(cls, dimension):
        assert dimension > 0, "The dimension must be a positive number"
        dim_range = np.arange(2, dimension + 1)

        def _gen_sample_(i, n):
            coeff = int(special.binom(n, i))
            return np.random.random_integers(low=0, high=coeff)

        vgen_sample_ = np.vectorize(_gen_sample_)
        subsets_counts = vgen_sample_(dim_range, dimension)

        while (np.sum(subsets_counts) == 0):
            subsets_counts = vgen_sample_(dim_range, dimension)

        z = zip(dim_range, subsets_counts)
        res = []
        hashs = []
        for t in z:
            if t[1] > 0:
                for j in np.arange(t[1]):
                    tmp_ = frozenset(np.random.choice(dimension, t[0], replace=False))
                    hash_ = hash(tmp_)
                    while hash_ in hashs:
                        tmp_ = frozenset(np.random.choice(dimension, t[0], replace=False))
                        hash_ = hash(tmp_)
                    res.append(tmp_)
                    hashs.append(hash_)

        res = np.array(res)

        lambdas_mo_h1_ind = np.arange(res.size)
        # From Crepey => simregr-jumps-SUBMITTED
        vlambdas_mo_h1_fun = np.vectorize(lambda i: 2 * 0.001 / (1 + i))
        lambdas_mo_h1 = vlambdas_mo_h1_fun(lambdas_mo_h1_ind)

        one_dim_range = np.arange(dimension)
        vfrozenset = np.vectorize(lambda i: frozenset([i]))
        one_dim_set = vfrozenset(one_dim_range)

        lambdas_mo_eq1_ind = np.arange(dimension)
        # From Crepey => simregr-jumps-SUBMITTED
        vlambdas_mo_eq1_fun = np.vectorize(lambda i: 0.0001 * (200 - i))
        lambdas_mo_eq1 = vlambdas_mo_eq1_fun(lambdas_mo_eq1_ind)

        subsets = np.append(res, one_dim_set)
        lambdas = np.append(lambdas_mo_h1, lambdas_mo_eq1)

        return subsets, lambdas


class StepWiseIntensitiesMarshallOlkinCopula(object):
    __max_tau = 1000

    def __init__(self, subsets, hazard_rates, pillars):
        self.__subsets = np.array(subsets)
        if self.__subsets.ndim == 1:
            self.__max_index = max([max(x) for x in self.__subsets])
            self.__subsets = np.array([self.__subsets]).T
        else:
            raise ValueError("The subsets dimension number must be 1")

        self.__pills = np.array(pillars)
        if self.__pills.ndim == 1:
            self.__pills = np.tile(self.pillars, (self.__subsets.size, 1))

        if self.__pills.shape[0] != self.__subsets.shape[0]:
            raise ValueError("The pillars dimension do not match with the number of subsets.")

        self.__hzrd_rates_mat = np.array(hazard_rates)
        h_subset_nb, h_pills_nb = self.__hzrd_rates_mat.shape
        if h_subset_nb == self.__pills.shape[1] and h_pills_nb == self.__subsets.shape[0]:
            self.__hzrd_rates_mat = self.__hzrd_rates_mat.T

        h_subset_nb, h_pills_nb = self.__hzrd_rates_mat.shape
        if h_subset_nb != self.__subsets.shape[0] or h_pills_nb != self.__pills.shape[1]:
            raise ValueError("The dimension do not match.")

        self.__models = [StepwiseConstantIntensity(pills, hz)
                         for hz, pills in zip(self.__hzrd_rates_mat, self.__pills)]

    @property
    def subsets(self):
        return np.array(self.__subsets, copy=True)

    @property
    def intensities(self):
        return np.array(self.__hzrd_rates_mat, copy=True)

    @property
    def pillars(self):
        return np.array(self.__pills, copy=True)

    @property
    def models(self):
        return np.array(self.__models, copy=True)

    def get_indexes_including(self, index):
        if index > self.__max_index:
            raise ValueError("The obligor_index must be lower than %s, given: %s"%(self.__max_index, index))

        res = []
        for (ii, set) in enumerate(self.subsets):
            if index in set[0]:
                res.append(ii)

        return res

    @staticmethod
    def __objective(models, exp_rvs, max_tau):
        res = []

        for m, e in zip(models, exp_rvs):
            f = lambda t: e + m.log_survival_proba(t)

            # No need to call the routine as it will fail because
            # f(0) is positive and f(max_tau) is also positive
            # thus the Brent routine is not necessary
            tau = None
            if f(max_tau) >= 0:
                tau = max_tau
            else:
                tau, result = brentq(f, 0, max_tau, full_output=True)
                if not result.converged:
                    tau = max_tau
            res.append(tau)

        return np.array(res)

    def generate_default_times(self, obligor_index=None, subsets_indexes=None , number=1, exp_rvs=None):
        if obligor_index is None and subsets_indexes is None:
            raise ValueError("Both obligor_index and subsets_indexes cannot be None at the same time")

        indexes = self.get_indexes_including(obligor_index) if obligor_index is not None else subsets_indexes
        models = self.models[indexes]

        if exp_rvs is not None:
            rvs = np.array(exp_rvs)
            if rvs.shape[1] != len(indexes):
                raise ValueError("The exp rv don't have the good columns size. "
                                 "Given %s, expected %s"%(rvs.shape[1], len(indexes)))
        else:
            rvs = expon.rvs(size=[number, len(indexes)])

        res = np.zeros(rvs.shape)
        for ii, exps in enumerate(rvs):
            res[ii, :] = self.__objective(models, exps, self.__max_tau)

        return res

    def gamma(self, subset_index, t):
        if subset_index >= self.subsets.shape[0]:
            raise ValueError("The subset_index must be leq than %s, "
                             "given: %s"%(self.subsets.shape[0], subset_index))

        model = self.models[subset_index]
        index = np.searchsorted(model.pillars, t, side='left') - 1
        index = np.minimum(index, model.intensities.size - 1)

        return model.intensities[index]

    def tot_gamma(self, t, obligor_index=None, subsets_indexes=None):
        if obligor_index is None and subsets_indexes is None:
            raise ValueError("Both obligor_index and subsets_indexes cannot be None at the same time")

        indexes = subsets_indexes
        if obligor_index is not None:
            indexes = self.get_indexes_including(obligor_index)

        models = self.models[indexes]

        gamma = 0.
        for m in models:
            if t == 0.:
                index = 0
            else:
                index = np.searchsorted(m.pillars, t, side='left') - 1
                index = np.minimum(index, m.intensities.size - 1)

            gamma += m.intensities[index]

        return gamma

    def tot_survival_proba(self, t, obligor_index=None, subsets_indexes=None):
        if obligor_index is None and subsets_indexes is None:
            raise ValueError("Both obligor_index and subsets_indexes cannot be None at the same time")

        indexes = subsets_indexes
        if obligor_index is not None:
            indexes = self.get_indexes_including(obligor_index)

        models = self.models[indexes]

        proba = 1.
        for m in models:
            proba *= m.survival_proba(t)

        return proba
