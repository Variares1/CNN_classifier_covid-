import matplotlib.pyplot as plt

def hist_plot(values, labels, metrics, title, bar_width=0.4):
    if type(values[0]) == list:
        bar_list = [plt.bar(range(len(labels)), values[0], width=bar_width)]
        i = 1
        for value in values[1:]:
            bar_list.append(plt.bar([x + bar_width * i for x in range(len(labels))], value, width=bar_width))
            i += 1
        plt.xticks([r + bar_width / len(values) for r in range(len(labels))], labels, rotation=45)
        plt.legend(bar_list, metrics, loc='best', bbox_to_anchor=(1, 0.5))
    else:
        plt.bar(range(len(labels)), values)
        plt.ylabel(metrics[0])
        plt.xticks(range(len(labels)), labels)
    plt.xlabel("algorithme")
    plt.title(title)

    plt.figure(figsize=(8, 6))
    plt.rcParams['figure.dpi'] = 100
    plt.show()

def compare_curve_plot(all_metric_dict,list_dict_1, list_dict_2, x_label, y_label_1, color_1, y_label_2, color_2):
    plot1 = plt.plot(all_metric_dict.keys(), list_dict_1, 'o-', label=y_label_1, color=color_1)
    plt.xticks(rotation=40)
    plt.xlabel(x_label)
    plt.ylabel(y_label_1, color=color_1)
    plt2 = plt.twinx()
    plot2 = plt2.plot(all_metric_dict.keys(), list_dict_2, 'o-', label=y_label_2, color=color_2)
    plt2.set_ylabel(y_label_2, color=color_2)

    allplot = plot1 + plot2
    alllabels = [l.get_label() for l in allplot]
    plt.legend(allplot, alllabels, loc=0)
    plt.title(y_label_1+'/'+y_label_2)
    plt.show()

def stock_results_dictionnary(all_metric_dict):
    list_dict = {}

    for algo in all_metric_dict.keys():
        for sample in ('train', 'test'):
            for score in all_metric_dict[algo][sample].keys():
                for info in ('mean', 'std'):
                    key = '%s_%s_%s_list' % (sample, score, info)
                    if key not in list_dict.keys():
                        list_dict[key] = []
                    list_dict[key].append(all_metric_dict[algo][sample][score][info])
        for value in all_metric_dict[algo]:
            key = '%s' % (value)
            if key not in list_dict.keys():
                list_dict[key] = []
            list_dict[key].append(all_metric_dict[algo][value])
    return list_dict
