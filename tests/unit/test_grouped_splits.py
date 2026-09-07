from graph_attention.data import make_grouped_split_manifest


def test_grouped_split_keeps_source_snapshots_indivisible() -> None:
    sample_ids = tuple(
        f"slice-{group}-{slice_index}" for group in "abcdef" for slice_index in range(4)
    )
    group_ids = tuple(group for group in "abcdef" for _ in range(4))

    split = make_grouped_split_manifest(
        sample_ids,
        group_ids,
        seed=42,
        train_ratio=0.5,
        validation_ratio=1.0 / 6.0,
    )

    group_by_sample = dict(zip(sample_ids, group_ids, strict=True))
    train_groups = {group_by_sample[sample] for sample in split.train_ids}
    validation_groups = {group_by_sample[sample] for sample in split.validation_ids}
    test_groups = {group_by_sample[sample] for sample in split.test_ids}

    assert len(train_groups) == 3
    assert len(validation_groups) == 1
    assert len(test_groups) == 2
    assert train_groups.isdisjoint(validation_groups)
    assert train_groups.isdisjoint(test_groups)
    assert validation_groups.isdisjoint(test_groups)
    assert set(split.all_ids) == set(sample_ids)


def test_grouped_split_is_deterministic_for_same_seed() -> None:
    sample_ids = tuple(f"sample-{index}" for index in range(10))
    group_ids = tuple(f"group-{index // 2}" for index in range(10))

    first = make_grouped_split_manifest(
        sample_ids, group_ids, seed=7, train_ratio=0.6, validation_ratio=0.2
    )
    second = make_grouped_split_manifest(
        sample_ids, group_ids, seed=7, train_ratio=0.6, validation_ratio=0.2
    )

    assert first == second
