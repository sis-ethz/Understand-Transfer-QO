import matplotlib.pyplot as plt
import os


class DecisionVisualizer:

    def __init__(self):
        pass

    def plot_2d_optimal_decision_with_importance(self, all_results, title='', filename='', x_label='selectivity on indexed attr', y_label='left cardinality ratio', base_dir='./'):

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

        plt.clf()
        plt.figure()
        ax = plt.gca()

        for f in all_results:

            sel, random_size, nl_idx_scan_cost, nl_seq_scan_cost, hash_idx_scan_cost, hash_seq_scan_cost, merge_idx_scan_cost, merge_seq_scan_cost = f[
                'visualization_features']

            m_cost_list = [nl_idx_scan_cost, nl_seq_scan_cost, hash_idx_scan_cost,
                           hash_seq_scan_cost, merge_idx_scan_cost, merge_seq_scan_cost]

            min_cost = min(m_cost_list)
            m_cost_list.remove(min_cost)

            importance = (min(m_cost_list) - min_cost) / min_cost

            if merge_idx_scan_cost == min_cost:
                merge_idx_scan_x.append(sel)
                merge_idx_scan_y.append(random_size)
                merge_idx_scan_imp.append(importance)
                # print("5")
            elif merge_seq_scan_cost == min_cost:
                merge_seq_scan_x.append(sel)
                merge_seq_scan_y.append(random_size)
                merge_seq_scan_imp.append(importance)

            elif hash_idx_scan_cost == min_cost:
                hash_idx_scan_x.append(sel)
                hash_idx_scan_y.append(random_size)
                hash_idx_scan_imp.append(importance)
                # print("1")
            elif hash_seq_scan_cost == min_cost:
                hash_seq_scan_x.append(sel)
                hash_seq_scan_y.append(random_size)
                hash_seq_scan_imp.append(importance)
                # print("2")
            elif nl_idx_scan_cost == min_cost:
                nl_idx_scan_x.append(sel)
                nl_idx_scan_y.append(random_size)
                nl_idx_scan_imp.append(importance)
                # print("3")
            elif nl_seq_scan_cost == min_cost:
                nl_seq_scan_x.append(sel)
                nl_seq_scan_y.append(random_size)
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

        # for x, y in zip(hash_idx_scan_x, hash_idx_scan_y):
        #     circle = plt.Circle((x, y), 0.01, color='b', fill=True)
        #     ax.add_artist(circle)

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
        if not filename:
            filename = title

        if not os.path.isdir(base_dir):
            os.system(f'mkdir {base_dir}')

        plt.savefig(os.path.join(base_dir, f'{filename}_all_op.png'))
