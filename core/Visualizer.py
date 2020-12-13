import matplotlib.pyplot as plt
import os
import numpy as np

def plot_2d_decision_boundaries(clf, X, X_costs, y, title='', filename=None, plot_original_data_class=False):       
    
    decision_boundary(clf, X, y, background_alpha=0.1, plot_original_data=plot_original_data_class)
    X_importance_viz = np.concatenate((X,X_costs), axis=1)
    if not plot_original_data_class:
        plot_2d_optimal_decision_with_importance(list(map(tuple, X_importance_viz)), reset_fig=False, title=title, filename=filename)
    plt.show()

def plot_2d_optimal_decision_with_importance(all_results, reset_fig=True, title='', filename='', x_label='selectivity on indexed attr', y_label='left cardinality ratio', base_dir='./'):

    hash_idx_scan_x = []
    hash_idx_scan_y = []
    hash_idx_scan_imp = []

    hash_seq_scan_x = []
    hash_seq_scan_y = []
    hash_seq_scan_imp = []

    nl_idx_scan_x = []
    nl_idx_scan_y = []
    nl_idx_scan_imp = []

    nl_seq_scan_x = []
    nl_seq_scan_y = []
    nl_seq_scan_imp = []

    merge_idx_scan_x = []
    merge_idx_scan_y = []
    merge_idx_scan_imp = []

    merge_seq_scan_x = []
    merge_seq_scan_y = []
    merge_seq_scan_imp = []

    if reset_fig:
        plt.clf()
        plt.figure(figsize=(8, 6), dpi=300)

    ax = plt.gca()

    for f in all_results:

        # x, y, hash_idx_scan_cost, hash_seq_scan_cost, nl_idx_scan_cost, nl_seq_scan_cost, merge_idx_scan_cost, merge_seq_scan_cost = f[
        #     'visualization_features']
        x, y, hash_idx_scan_cost, hash_seq_scan_cost, nl_idx_scan_cost, nl_seq_scan_cost, merge_idx_scan_cost, merge_seq_scan_cost = f
        m_cost_list = [nl_idx_scan_cost, nl_seq_scan_cost, hash_idx_scan_cost,
                       hash_seq_scan_cost, merge_idx_scan_cost, merge_seq_scan_cost]

        min_cost = min(m_cost_list)
        m_cost_list.remove(min_cost)

        importance = (min(m_cost_list) - min_cost) / min_cost

        if merge_idx_scan_cost == min_cost:
            merge_idx_scan_x.append(x)
            merge_idx_scan_y.append(y)
            merge_idx_scan_imp.append(importance)
            # print("5")
        elif merge_seq_scan_cost == min_cost:
            merge_seq_scan_x.append(x)
            merge_seq_scan_y.append(y)
            merge_seq_scan_imp.append(importance)

        elif hash_idx_scan_cost == min_cost:
            hash_idx_scan_x.append(x)
            hash_idx_scan_y.append(y)
            hash_idx_scan_imp.append(importance)
            # print("1")
        elif hash_seq_scan_cost == min_cost:
            hash_seq_scan_x.append(x)
            hash_seq_scan_y.append(y)
            hash_seq_scan_imp.append(importance)
            # print("2")
        elif nl_idx_scan_cost == min_cost:
            nl_idx_scan_x.append(x)
            nl_idx_scan_y.append(y)
            nl_idx_scan_imp.append(importance)
            # print("3")
        elif nl_seq_scan_cost == min_cost:
            nl_seq_scan_x.append(x)
            nl_seq_scan_y.append(y)
            nl_seq_scan_imp.append(importance)
            # print("4")

            # print("6")
        # print([nl_idx_scan_cost, nl_seq_scan_cost, hash_idx_scan_cost,
        #        hash_seq_scan_cost, merge_idx_scan_cost, merge_seq_scan_cost])
    # print(m_color(hash_idx_scan_imp))

    cm = plt.cm.get_cmap('RdYlBu')
    # sc = plt.scatter(xy, xy, c=z, vmin=0, vmax=20, s=35, cmap=cm)
    # plt.colorbar(sc)

    pt = plt.scatter(hash_idx_scan_x, hash_idx_scan_y, marker='+', c=hash_idx_scan_imp,
                     label='hash join + index scan')

    plt.scatter(hash_seq_scan_x, hash_seq_scan_y, marker='*', c=hash_seq_scan_imp,
                label='hash join + seq scan')

    plt.scatter(nl_idx_scan_x, nl_idx_scan_y, marker='o', c=nl_idx_scan_imp,
                label='nested loop + index scan')

    plt.scatter(nl_seq_scan_x, nl_seq_scan_y, marker='x', c=nl_seq_scan_imp,
                label='nested loop + seq scan')

    plt.scatter(merge_idx_scan_x, merge_idx_scan_y, marker='v', c=merge_idx_scan_imp,
                label='merge join + index scan')

    plt.scatter(merge_seq_scan_x, merge_seq_scan_y, marker='^', c=merge_seq_scan_imp,
                label='merge join + seq scan')

    plt.colorbar()

    plt.legend(loc="upper left")
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # print("filename: ", filename)
    # print("title: ", title)

    if filename is None:
        # plt.show()
        pass
    else:
        if not os.path.isdir(base_dir):
            os.system(f'mkdir {base_dir}')
        plt.savefig(os.path.join(base_dir, f'{filename}_all_op.png'))


def plot_3d_optimal_decision_with_importance(all_results, reset_fig=True, title='', filename='', labels=['', '', ''], base_dir='./'):
    print("Pls view the 3d plot in jupyter notebook")
    # Import dependencies
    import plotly
    import plotly.graph_objs as go
    # Configure Plotly to be rendered inline in the notebook.
    plotly.offline.init_notebook_mode()

    hash_idx_scan_x = []
    hash_idx_scan_y = []
    hash_idx_scan_z = []
    hash_idx_scan_imp = []

    hash_seq_scan_x = []
    hash_seq_scan_y = []
    hash_seq_scan_z = []
    hash_seq_scan_imp = []

    nl_idx_scan_x = []
    nl_idx_scan_y = []
    nl_idx_scan_z = []
    nl_idx_scan_imp = []

    nl_seq_scan_x = []
    nl_seq_scan_y = []
    nl_seq_scan_z = []
    nl_seq_scan_imp = []

    merge_idx_scan_x = []
    merge_idx_scan_y = []
    merge_idx_scan_z = []
    merge_idx_scan_imp = []

    merge_seq_scan_x = []
    merge_seq_scan_y = []
    merge_seq_scan_z = []
    merge_seq_scan_imp = []

    for f in all_results:

        # sel, y, hash_idx_scan_cost, hash_seq_scan_cost, nl_idx_scan_cost, nl_seq_scan_cost, merge_idx_scan_cost, merge_seq_scan_cost = f[
        #     'visualization_features']
        x, y, z, hash_idx_scan_cost, hash_seq_scan_cost, nl_idx_scan_cost, nl_seq_scan_cost, merge_idx_scan_cost, merge_seq_scan_cost = f
        m_cost_list = [nl_idx_scan_cost, nl_seq_scan_cost, hash_idx_scan_cost,
                       hash_seq_scan_cost, merge_idx_scan_cost, merge_seq_scan_cost]

        min_cost = min(m_cost_list)
        m_cost_list.remove(min_cost)

        importance = (min(m_cost_list) - min_cost) / min_cost

        if merge_idx_scan_cost == min_cost:
            merge_idx_scan_x.append(x)
            merge_idx_scan_y.append(y)
            merge_idx_scan_z.append(z)
            merge_idx_scan_imp.append(importance)
        elif merge_seq_scan_cost == min_cost:
            merge_seq_scan_x.append(x)
            merge_seq_scan_y.append(y)
            merge_seq_scan_z.append(z)
            merge_seq_scan_imp.append(importance)
        elif hash_idx_scan_cost == min_cost:
            hash_idx_scan_x.append(x)
            hash_idx_scan_y.append(y)
            hash_idx_scan_z.append(z)
            hash_idx_scan_imp.append(importance)
        elif hash_seq_scan_cost == min_cost:
            hash_seq_scan_x.append(x)
            hash_seq_scan_y.append(y)
            hash_seq_scan_z.append(z)
            hash_seq_scan_imp.append(importance)
        elif nl_idx_scan_cost == min_cost:
            nl_idx_scan_x.append(x)
            nl_idx_scan_y.append(y)
            nl_idx_scan_z.append(z)
            nl_idx_scan_imp.append(importance)
        elif nl_seq_scan_cost == min_cost:
            nl_seq_scan_x.append(x)
            nl_seq_scan_y.append(y)
            nl_seq_scan_z.append(z)
            nl_seq_scan_imp.append(importance)

    xs = [hash_idx_scan_x, hash_seq_scan_x, nl_idx_scan_x,
          nl_seq_scan_x, merge_idx_scan_x, merge_seq_scan_x]
    ys = [hash_idx_scan_y, hash_seq_scan_y, nl_idx_scan_y,
          nl_seq_scan_y, merge_idx_scan_y, merge_seq_scan_y]
    zs = [hash_idx_scan_z, hash_seq_scan_z, nl_idx_scan_z,
          nl_seq_scan_z, merge_idx_scan_z, merge_seq_scan_z]
    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow']
    trace = []

    for i in range(len(xs)):
        trace.append(go.Scatter3d(
            x=xs[i],  # <-- Put your data instead
            y=ys[i],  # <-- Put your data instead
            z=zs[i],  # <-- Put your data instead
            mode='markers',
            marker={
                'size': 2,
                'opacity': 1,
                'color': colors[i]
            }
        ))
    # layout = go.Layout(
    #     margin={'l': 0, 'r': 0, 'b': 0, 't': 0}
    # )
    data = trace
    plot_figure = go.Figure(data=data)
    plot_figure.update_layout(
        title=title,
        # xaxis_title=labels[0],
        # yaxis_title=labels[1],
        # # zaxis_title=labels[2],
    )

    plotly.offline.iplot(plot_figure)


def decision_boundary(model, X, y, background_alpha=0.2, reset_fig=True, show=True, plot_original_data=False):
    from matplotlib.colors import ListedColormap, LinearSegmentedColormap
    import matplotlib as mpl

    class m_normalize():
        def __init__(self,):
            super(m_normalize, self).__init__()
            self.vmax = 1
            self.vmin = 0

        def process_value(value):
            is_scalar = not np.iterable(value)
            return value, is_scalar
        
        def __call__(self, value, clip=None):
            return value

        def inverse(self, value):
            return value

        def autoscale_None(self, A):
            pass

    cmap = LinearSegmentedColormap.from_list('map', [(0.0, 'b'), (1/5, 'g'), (2/5, 'r'), (3/5, 'c'), (4/5, 'm'), (1.0, 'y')])
    plot_step = 0.01  # fine step width for decision surface contours
    # plot_step_coarser = 0.01  # step widths for coarse classifier guesses
    RANDOM_SEED = 1  # fix the seed on each iteration

    if reset_fig:
        plt.clf()
        plt.figure(figsize=(8, 6), dpi=300)

    # Now plot the decision boundary using a fine mesh as input to a
    # filled contour plot

    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()

    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))

    # Plot either a single DecisionTreeClassifier or alpha blend the
    # decision surfaces of the ensemble of classifiers
    if True:  # not isinstance(model, random_forest):
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        Z = Z / 5
        # cs = plt.contourf(xx, yy, Z, cmap=ListedColormap(
        #     ['b', 'g', 'r', 'c', 'm', 'y']), norm=m_normalize(), alpha=background_alpha)
        cs = plt.contourf(xx, yy, Z, cmap=cmap, alpha=background_alpha)

    if plot_original_data:
        plt.scatter(X[:, 0], X[:, 1], c=model.predict(X), cmap=cmap, s=20)


def visualize_decision_tree(tree, feature_names, filled=True, font_size=10, figure_size=(12, 12), show_figure=True):
    from sklearn import tree
    plt.clf()
    fig, ax = plt.subplots(figsize=figure_size, dpi=300)
    tree.plot_tree(tree, feature_names=feature_names,
                   filled=filled, fontsize=font_size)
    if show_figure:
        plt.show()
