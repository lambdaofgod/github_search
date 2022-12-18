import paperswithcode as pwc

client = pwc.PapersWithCodeClient()


def get_areas(key="id"):
    return [dict(a)[key] for a in client.area_list().results]


def get_area_tasks(area):
    return client.area_task_list(area, items_per_page=1000).results
