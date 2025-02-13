def replace_items_with_sub_list_and_idxs(
    items: list, sub_items: list[list], idxs: list[int], func: callable = None
) -> list:
    res = []
    for i, item in enumerate(items):
        if i in idxs:
            res.extend(sub_items[idxs.index(i)])
        else:
            if func:
                res.append(func(item))
            else:
                res.append(item)

    return res
