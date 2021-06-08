from multiprocessing import Pool
import time
import random
from itertools import compress
import matplotlib.pyplot as plt
from seirsplus.models import *
from seirsplus.FARZ import *
import networkx
import numpy as np
import scipy.stats as stats
import pickle
import ipdb


class SEIRNetExperiment:
    def __init__(self, config_dict={}, init_network=None):

        self.stats_net = None
        self.beta = config_dict.get('beta', 0.155)
        self.sigma = config_dict.get('sigma', 1 / 5.2)
        self.gamma = config_dict.get('gamma', 1 / 12.39)
        self.init_I = config_dict.get('init_I', 20)

        # configuration of the network
        if init_network:
            self.network = init_network
            self.nodes = init_network.number_of_nodes()
            self.edges = init_network.number_of_edges(
            ) / init_network.number_of_nodes()
        else:
            if 'network_family' in config_dict:
                self.network_family = config_dict["network_family"]
            else:
                raise RuntimeError(
                    f'Please provide the network family in the config dictionary or an initialised network'
                )

            self.nodes = config_dict.get('nodes', 1000)
            self.edges = config_dict.get('edges', 10)

            # all of these hyperparams are only needed if network is to be initialised
            if self.network_family == 'farz':
                # exclusive of farz family
                self.k = config_dict.get("k", 1)
                self.a = config_dict.get("a", 0.5)
                self.g = config_dict.get("g", 0.5)
                self.b = config_dict.get("b", 0.8)
                self.r = config_dict.get("r", 1)
                self.q = config_dict.get("q", 0.5)
                self.t = config_dict.get("t", 0.0)
                self.e = config_dict.get("e", 0.0000001)
            elif self.network_family in ['powerlaw', 'smallworld']:
                self.p = config_dict.get("p", 0.0)

            self.initialise_network()

        # contact = different individuals have different network sizes but same amount of contact (equal budget)
        # connect = different individuals have different network sizes and also different connectivity to their network (unequal budget)
        self.budget = config_dict.get('budget', 'unequal')


    def initialise_network(self):
        desired_nedges = int(self.nodes * (self.edges / 2))
        # some graph generators do not return exactly the same number of edges so
        # sometimes we need to complete it with a very small amount of random connections
        if self.network_family == 'lattice':
            self.network = networkx.random_regular_graph(d=self.edges,
                                                         n=self.nodes)
        elif self.network_family == 'erdosrenyi':
            self.network = networkx.gnm_random_graph(n=self.nodes,
                                                     m=int(desired_nedges))
        elif self.network_family == 'barabasi':
            self.network = networkx.barabasi_albert_graph(n=self.nodes,
                                                          m=self.edges // 2)
            complete_graph(self.network, desired_nedges)
        elif self.network_family == 'powerlaw':
            self.network = networkx.powerlaw_cluster_graph(n=self.nodes,
                                                           m=self.edges // 2,
                                                           p=self.p)
            complete_graph(self.network, desired_nedges)
        elif self.network_family == 'smallworld':
            self.network = networkx.connected_watts_strogatz_graph(
                n=self.nodes, k=self.edges, p=self.p)
        elif self.network_family == 'farz':
            setting = {
                "n": self.nodes,
                "k": self.k,
                "m": self.edges // 2,
                "alpha": self.a,
                "gamma": self.g,
                "beta": self.b,
                "phi": 1,
                'r': 1 + round(self.r * self.k),
                'q': self.q,
                "epsilon": self.e,
                'directed': False,
                'weighted': False,
                'b': self.t,
            }
            self.network, _ = generate(vari=None,
                                       arange=None,
                                       repeat=1,
                                       path='.',
                                       net_name='network',
                                       format='gml',
                                       farz_params=setting.copy())
            complete_graph(self.network, desired_nedges)
        else:
            raise RuntimeError(
                f'Wrong family of networks to create: {self.network_family}')

    def compute_stats_net(self):
        degree_sequence = sorted([d for n, d in self.network.degree()],
                                 reverse=True)  # degree sequence
        stats_net = {}
        # compute degree measures
        stats_net["avg_degree"] = np.mean(degree_sequence)
        stats_net["std_degree"] = np.std(degree_sequence)
        stats_net["max_degree"] = np.max(degree_sequence)
        stats_net["min_degree"] = np.min(degree_sequence)
        stats_net["kur_degree"] = stats.kurtosis(degree_sequence)
        stats_net["skew_degree"] = stats.skew(degree_sequence)
        count = np.bincount(degree_sequence)
        stats_net["entropy_degree"] = stats.entropy(count[count != 0] /
                                                    sum(count))


        # some metrics don't work without a connected graph
        stats_net["is_con"] = int(networkx.is_connected(self.network))
        if stats_net["is_con"]:
            ecc = networkx.eccentricity(self.network)
            ecc = np.array(list(ecc.values()))
            stats_net["avg_eccentricity"] = ecc.mean()
            stats_net["std_eccentricity"] = ecc.std()
            stats_net["skew_eccentricity"] = stats.skew(ecc)
            stats_net["diameter"] = networkx.diameter(self.network)
            stats_net["radius"] = networkx.radius(self.network)
            stats_net["w_ind"] = networkx.wiener_index(self.network)
        else:
            stats_net["avg_eccentricity"] = float("NaN")
            stats_net["std_eccentricity"] = float("NaN")
            stats_net["skew_eccentricity"] = float("NaN")
            stats_net["diameter"] = float("NaN")
            stats_net["radius"] = float("NaN")
            stats_net["w_ind"] = float("NaN")

        clos_c = networkx.closeness_centrality(self.network)
        clos_c = np.array(list(clos_c.values()))
        bet_c = networkx.betweenness_centrality(self.network)
        bet_c = np.array(list(bet_c.values()))
        stats_net["avg_clos"] = np.mean(clos_c)
        stats_net["std_clos"] = np.std(clos_c)
        stats_net["skew_clos"] = stats.skew(clos_c)
        stats_net["avg_betw"] = np.mean(bet_c)
        stats_net["std_betw"] = np.std(bet_c)
        stats_net["skew_betw"] = stats.skew(bet_c)

        # connectivity measures
        stats_net["node_conn"] = networkx.node_connectivity(self.network)
        stats_net["clust"] = networkx.average_clustering(self.network)
        stats_net["trans"] = networkx.transitivity(self.network)
        stats_net[
            "assort_corr"] = networkx.degree_pearson_correlation_coefficient(
                self.network)
        stats_net["is_con"] = int(networkx.is_connected(self.network))
        stats_net["con_comp"] = networkx.number_connected_components(
            self.network)
        stats_net["loc_eff"] = networkx.local_efficiency(self.network)
        stats_net["glb_eff"] = networkx.global_efficiency(self.network)

        # spectral measures
        stats_net["alg_conn"] = networkx.algebraic_connectivity(
            self.network, method='lanczos')
        stats_net["spectral_radius"] = np.max(
            networkx.adjacency_spectrum(self.network).real)

        self.stats_net = stats_net
        return stats_net

    def simulate(self, p=0.0, init_model=None):
        if init_model is not None:
            simsteps_previous = len(init_model.numS)

            inf_previous = self.nodes - init_model.numS[-1]
            if (inf_previous + self.init_I) >= self.nodes:
                inf = self.nodes - inf_previous
                peak = self.nodes - inf_previous
                r0 = 1.0
                simsteps = 0
                model = {}
            else:
                model = init_model
                model.X = add_infected(init_model.X, ninfected=self.init_I)
                model.run(T=13, verbose=False)
                r0 = (self.nodes - model.numS[-1] - inf_previous) / self.init_I
                model.run(T=1000, verbose=False)
                inf = self.nodes - model.numS[-1] - inf_previous
                peak = max(model.numI[simsteps_previous:])
                simsteps = len(model.numS) - simsteps_previous
        else:
            model = SEIRSNetworkModel(G=self.network,
                                          beta=self.beta,
                                          sigma=self.sigma,
                                          gamma=self.gamma,
                                          initI=self.init_I,
                                          p=p,
                                          budget=self.budget)

            model.run(T=13, verbose=False)
            r0 = (self.nodes - model.numS[-1]) / self.init_I
            # run the sim until it stops or 1000 steps are achieved
            model.run(T=1000, verbose=False)
            # compute infected, peak and R0
            inf = self.nodes - model.numS[-1]
            peak = max(model.numI)
            simsteps = len(model.numS)

        return model, inf, peak, r0, simsteps


def random_edge(graph):
    '''
    Create a new random edge
    :param graph: networkx graph
    :return: networkx graph
    '''
    #edges = list(graph.edges)
    nonedges = list(networkx.non_edges(graph))
    # random edge choice
    #chosen_edge = random.choice(edges)
    chosen_nonedge = random.choice(nonedges)
    graph.add_edge(chosen_nonedge[0], chosen_nonedge[1])


def replace_random(graph, ratio=0.1):
    '''
    Replace a percentage of the edges with random connections
    :param graph: networkx graph
    :return: networkx graph
    '''
    n = graph.number_of_nodes()
    to_replace = int(np.floor(n * ratio))
    for i in range(to_replace):
        edges = list(graph.edges)
        #nonedges = list(networkx.non_edges(graph))
        # random edge choice
        chosen_edge = random.choice(edges)
        # remove one edge after (i.e. replace one edge)
        graph.remove_edge(chosen_edge[0], chosen_edge[1])
        random_edge(graph)


def complete_graph(graph, desired_edges):
    '''
    :param graph: networkx graph
    :return: networkx graph
    '''
    number_to_add = desired_edges - graph.number_of_edges()
    for i in range(number_to_add):
        random_edge(graph)


def add_infected(individuals, ninfected=20):
    for i in range(ninfected + 1):
        # find susceptible
        suscept = np.where(individuals == 1)
        chosen = random.choice(suscept[0])
        individuals[chosen] = 3
    return individuals


def run_sim(exp, reps=10, compute_stats=False):
    """
    1. create different sets of realistic networks.
    2. for each network:
        1. run seir simulation 10 times (compute average + std of infected, peak, simsteps, r0)
        2. compute stats of network (clustering, degree distribution, local efficiency...)
        3. reintroduce the virus and rerun simulation 10 times with reduced graph. ("How resilient is each network to reintroducing the virus"?)
        4. How resilient is each network to random changes in the network structure? (replace 10% of the network with random connections)
    """
    # normal stats
    inf = np.zeros(reps)
    peak = np.zeros(reps)
    R0 = np.zeros(reps)
    simsteps = np.zeros(reps)

    # perturbed net (replacing a percentage with random connections)
    inf_pert = np.zeros(reps)
    peak_pert = np.zeros(reps)
    R0_pert = np.zeros(reps)
    simsteps_pert = np.zeros(reps)

    # remaining net after first infection
    inf_pert_rem = np.zeros(reps)
    peak_pert_rem = np.zeros(reps)
    R0_pert_rem = np.zeros(reps)
    simsteps_pert_rem = np.zeros(reps)

    # perturbed net (replacing a percentage with random connections)
    inf_pert2 = np.zeros(reps)
    peak_pert2 = np.zeros(reps)
    R0_pert2 = np.zeros(reps)
    simsteps_pert2 = np.zeros(reps)

    # remaining net after first infection
    inf_pert_rem2 = np.zeros(reps)
    peak_pert_rem2 = np.zeros(reps)
    R0_pert_rem2 = np.zeros(reps)
    simsteps_pert_rem2 = np.zeros(reps)

    # remaining net after first infection
    inf_rem = np.zeros(reps)
    peak_rem = np.zeros(reps)
    R0_rem = np.zeros(reps)
    simsteps_rem = np.zeros(reps)

    # compute stats of my initial network
    if compute_stats:
        stats_orig = exp.compute_stats_net()
    else:
        stats_orig = {}
    # for saving all models
    model_dict = {}
    model_pert_dict = {}
    model_rem_dict = {}
    model_pert_dict2 = {}
    model_pert_rem_dict = {}
    model_pert_rem_dict2 = {}
    net_pert_dict = {}
    net_rem_dict = {}
    stats_pert = {}
    stats_rem = {}

    # repeat the simulations a number of times for the network
    for rep in range(reps):
        # run original net
        model_dict[rep], inf[rep], peak[rep], R0[rep], simsteps[
            rep] = exp.simulate()

        # running remaining net
        # simulate reintroducing the virus once its gone and herd immunity
        # is thought to be in place
        model_rem_dict[rep], inf_rem[rep], peak_rem[rep], R0_rem[
            rep], simsteps_rem[rep] = exp.simulate(init_model=model_dict[rep])

        # running perturbed net with p=0.1
        network_pert = exp.network.copy()
        exp_pert = SEIRNetExperiment(init_network=network_pert)
        model_pert_dict[rep], inf_pert[rep], peak_pert[rep], R0_pert[
            rep], simsteps_pert[rep] = exp_pert.simulate(p=0.1)

        # running second wave of perturbed net with p=0.1
        model_pert_rem_dict[rep], inf_pert_rem[rep], peak_pert_rem[
            rep], R0_pert_rem[rep], simsteps_pert_rem[rep] = exp_pert.simulate(
                init_model=model_pert_dict[rep])

        # running perturbed net with p=0.2
        exp_pert2 = SEIRNetExperiment(init_network=network_pert)
        model_pert_dict2[rep], inf_pert2[rep], peak_pert2[rep], R0_pert2[
            rep], simsteps_pert2[rep] = exp_pert2.simulate(p=0.2)

        # running second wave of perturbed net with p=0.2
        model_pert_rem_dict2[rep], inf_pert_rem2[rep], peak_pert_rem2[
            rep], R0_pert_rem2[rep], simsteps_pert_rem2[
                rep] = exp_pert2.simulate(init_model=model_pert_dict2[rep])

    stats_orig["mean_inf"] = np.mean(inf)
    stats_orig["std_inf"] = np.std(inf)
    stats_orig["mean_peak"] = np.mean(peak)
    stats_orig["std_peak"] = np.std(peak)
    stats_orig["mean_r0"] = np.mean(R0)
    stats_orig["std_r0"] = np.std(R0)
    stats_orig["mean_simsteps"] = np.mean(simsteps)
    stats_orig["std_simsteps"] = np.std(simsteps)

    stats_orig["mean_inf_rem"] = np.mean(inf_rem)
    stats_orig["std_inf_rem"] = np.std(inf_rem)
    stats_orig["mean_peak_rem"] = np.mean(peak_rem)
    stats_orig["std_peak_rem"] = np.std(peak_rem)
    stats_orig["mean_r0_rem"] = np.mean(R0_rem)
    stats_orig["std_r0_rem"] = np.std(R0_rem)
    stats_orig["mean_simsteps_rem"] = np.mean(simsteps_rem)
    stats_orig["std_simsteps_rem"] = np.std(simsteps_rem)

    stats_orig["mean_inf_pert"] = np.mean(inf_pert)
    stats_orig["std_inf_pert"] = np.std(inf_pert)
    stats_orig["mean_peak_pert"] = np.mean(peak_pert)
    stats_orig["std_peak_pert"] = np.std(peak_pert)
    stats_orig["mean_r0_pert"] = np.mean(R0_pert)
    stats_orig["std_r0_pert"] = np.std(R0_pert)
    stats_orig["mean_simsteps_pert"] = np.mean(simsteps_pert)
    stats_orig["std_simsteps_pert"] = np.std(simsteps_pert)

    stats_orig["mean_inf_pert2"] = np.mean(inf_pert2)
    stats_orig["std_inf_pert2"] = np.std(inf_pert2)
    stats_orig["mean_peak_pert2"] = np.mean(peak_pert2)
    stats_orig["std_peak_pert2"] = np.std(peak_pert2)
    stats_orig["mean_r0_pert2"] = np.mean(R0_pert2)
    stats_orig["std_r0_pert2"] = np.std(R0_pert2)
    stats_orig["mean_simsteps_pert2"] = np.mean(simsteps_pert2)
    stats_orig["std_simsteps_pert2"] = np.std(simsteps_pert2)

    stats_orig["mean_inf_pert_rem"] = np.mean(inf_pert_rem)
    stats_orig["std_inf_pert_rem"] = np.std(inf_pert_rem)
    stats_orig["mean_peak_pert_rem"] = np.mean(peak_pert_rem)
    stats_orig["std_peak_pert_rem"] = np.std(peak_pert_rem)
    stats_orig["mean_r0_pert_rem"] = np.mean(R0_pert_rem)
    stats_orig["std_r0_pert_rem"] = np.std(R0_pert_rem)
    stats_orig["mean_simsteps_pert_rem"] = np.mean(simsteps_pert_rem)
    stats_orig["std_simsteps_pert_rem"] = np.std(simsteps_pert_rem)

    stats_orig["mean_inf_pert_rem2"] = np.mean(inf_pert_rem2)
    stats_orig["std_inf_pert_rem2"] = np.std(inf_pert_rem2)
    stats_orig["mean_peak_pert_rem2"] = np.mean(peak_pert_rem2)
    stats_orig["std_peak_pert_rem2"] = np.std(peak_pert_rem2)
    stats_orig["mean_r0_pert_rem2"] = np.mean(R0_pert_rem2)
    stats_orig["std_r0_pert_rem2"] = np.std(R0_pert_rem2)
    stats_orig["mean_simsteps_pert_rem2"] = np.mean(simsteps_pert_rem2)
    stats_orig["std_simsteps_pert_rem2"] = np.std(simsteps_pert_rem2)

    return stats_orig, model_dict, model_rem_dict, net_rem_dict, stats_rem, model_pert_dict, model_pert_dict2, model_pert_rem_dict, model_pert_rem_dict2


def run_experiment(config_dict, iter):
    folder = '/mnt/DataVolume/maria/results/'
    exp = SEIRNetExperiment(config_dict)
    stats, models_orig, models_rem, net_rem, stats_rem, models_pert, models_pert2, models_pert_rem, models_pert_rem2 = run_sim(
        exp, reps=10, compute_stats=False)
    stats["iter"] = iter
    to_save = {**stats, **config_dict}

    to_save_keys = [i for i in to_save.keys()]
    to_save_values = [i for i in to_save.values()]

    config_string = '_'.join("{!s}={!r}".format(key, val)
                             for (key, val) in config_dict.items())
    config_string = config_string.replace('.', '')
    config_string = config_string.replace('\'', '')
    all_results_dict = {
        'network_orig': exp.network,
        'network_rem': net_rem,
        'stats': stats,
        'model': models_orig,
        'models_rem': models_rem,
        'models_pert': models_pert,
        'models_pert2': models_pert2,
        'models_pert_rem': models_pert_rem,
        'models_pert_rem2': models_pert_rem2,
        'config_dict': config_dict
    }
    with open(
            folder + 'saved_stats_model_' + config_string + '_iter_' +
            str(iter) + '.dictionary', 'wb') as config_dictionary_file:
        pickle.dump(all_results_dict, config_dictionary_file)


def run_unequal(config_dict, iter):
    # load previously generated social graphs and run the unequal budget there
    file = config_dict["file"]
    dir_results = '/mnt/DataVolume/maria/results_new/'
    if not os.path.isfile(dir_results + 'redone/' + file):
        pickle_in = open(dir_results + file, "rb")
        all_results_dict_old = pickle.load(pickle_in)
        exp = SEIRNetExperiment(
            init_network=all_results_dict_old['network_orig'])
        exp.budget = 'unequal'
        stats, _, _, _, _, _, _, _, _ = run_sim(exp,
                                                reps=10,
                                                compute_stats=False)
        all_results_dict_new = {'stats_unequal': stats}
        with open(dir_results + 'redone/' + file,
                  'wb') as config_dictionary_file:
            pickle.dump(all_results_dict_new, config_dictionary_file)


def run_unequal_pert(config_dict, iter):
    file = config_dict["file"]
    dir_results = '/mnt/DataVolume/maria/results_new/'

    pickle_in = open(dir_results + file, "rb")
    all_results_dict_old = pickle.load(pickle_in)

    exp = SEIRNetExperiment(init_network=all_results_dict_old['network_orig'])
    exp.personalisation = 'connect'
    rep = 0
    reps = 10

    # normal stats
    inf = np.zeros(reps)
    peak = np.zeros(reps)
    R0 = np.zeros(reps)
    simsteps = np.zeros(reps)

    # remaining net after first infection
    inf_rem = np.zeros(reps)
    peak_rem = np.zeros(reps)
    R0_rem = np.zeros(reps)
    simsteps_rem = np.zeros(reps)

    inf_pert = np.zeros(reps)
    peak_pert = np.zeros(reps)
    R0_pert = np.zeros(reps)
    simsteps_pert = np.zeros(reps)
    inf_pert_rem = np.zeros(reps)
    peak_pert_rem = np.zeros(reps)
    R0_pert_rem = np.zeros(reps)
    simsteps_pert_rem = np.zeros(reps)
    inf_pert2 = np.zeros(reps)
    peak_pert2 = np.zeros(reps)
    R0_pert2 = np.zeros(reps)
    simsteps_pert2 = np.zeros(reps)
    inf_pert_rem2 = np.zeros(reps)
    peak_pert_rem2 = np.zeros(reps)
    R0_pert_rem2 = np.zeros(reps)
    simsteps_pert_rem2 = np.zeros(reps)

    # resimulate for all models
    for rep in range(reps):
        model_aux, inf[rep], peak[rep], R0[rep], simsteps[rep] = exp.simulate(
            p=0.0)

        # running remaining net
        # simulate reintroducing the virus once its gone and herd immunity
        # is thought to be in place
        _, inf_rem[rep], peak_rem[rep], R0_rem[rep], simsteps_rem[
            rep] = exp.simulate(init_model=model_aux)

        model_aux, inf_pert[rep], peak_pert[rep], R0_pert[rep], simsteps_pert[
             rep] = exp.simulate(p=0.1)

        # running second wave of perturbed net with p=0.1
        _, inf_pert_rem[rep], peak_pert_rem[rep], R0_pert_rem[
            rep], simsteps_pert_rem[rep] = exp.simulate(init_model=model_aux)

        # running perturbed net with p=0.2
        model_aux, inf_pert2[rep], peak_pert2[rep], R0_pert2[
            rep], simsteps_pert2[rep] = exp.simulate(p=0.2)

        # running second wave of perturbed net with p=0.2
        _, inf_pert_rem2[rep], peak_pert_rem2[rep], R0_pert_rem2[
            rep], simsteps_pert_rem2[rep] = exp.simulate(init_model=model_aux)

    pickle_in2 = open(dir_results + 'redone/' + file, "rb")
    all_results_dict_old2 = pickle.load(pickle_in2)
    # recalculate averages
    stats = all_results_dict_old2['stats_unequal']
    stats["mean_inf"] = np.mean(inf)
    stats["std_inf"] = np.std(inf)
    stats["mean_peak"] = np.mean(peak)
    stats["std_peak"] = np.std(peak)
    stats["mean_r0"] = np.mean(R0)
    stats["std_r0"] = np.std(R0)
    stats["mean_simsteps"] = np.mean(simsteps)
    stats["std_simsteps"] = np.std(simsteps)
    stats["mean_inf_rem"] = np.mean(inf_rem)
    stats["std_inf_rem"] = np.std(inf_rem)
    stats["mean_peak_rem"] = np.mean(peak_rem)
    stats["std_peak_rem"] = np.std(peak_rem)
    stats["mean_r0_rem"] = np.mean(R0_rem)
    stats["std_r0_rem"] = np.std(R0_rem)
    stats["mean_simsteps_rem"] = np.mean(simsteps_rem)
    stats["std_simsteps_rem"] = np.std(simsteps_rem)


    all_results_dict_new = {
        'stats_unequal': stats,
        'config_dict': all_results_dict_old['config_dict']
    }
    with open(dir_results + 'redone/' + file, 'wb') as config_dictionary_file:
        pickle.dump(all_results_dict_new, config_dictionary_file)

