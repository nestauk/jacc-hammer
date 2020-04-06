def test_nearest_neighbour_matrix():
    # Test empty
    np.testing.assert_almost_equal(
        nearest_neighbour_matrix([None, None, None], shape=(2, 1)).todense(),
        ss.coo_matrix((2, 1)).todense(),
    )

    out = ss.coo_matrix(([0.5, 0.3, 0.2], ([0, 1, 1], [1, 0, 2])), shape=(4, 5)).tocsr()
    test_out = nearest_neighbour_matrix(
        [[(0, 1, 0.5)], None, [(1, 0, 0.3), (1, 2, 0.2)], None], shape=(4, 5)
    )
    np.testing.assert_almost_equal(test_out.todense(), out.todense())


def test_lok_sparse():
    def test(f):
        # Test empty
        np.testing.assert_almost_equal(
            f([[], []], shape=(2, 1)).todense(), ss.coo_matrix((2, 1)).todense()
        )

        out = ss.coo_matrix(([1, 1, 1], ([0, 2, 2], [1, 0, 2])), shape=(4, 5))
        test_out = f([[1], [], [0, 2], []], shape=(4, 5))
        np.testing.assert_almost_equal(test_out.todense(), out.todense())

    test(_lok_to_csr)
    test(_lok_to_coo)


def test_top_cosine_similarities():
    X_a = ss.csr_matrix(np.array([[0, 1, 0, 1], [1, 0, 0, 1], [0, 0, 1, 0]]))
    X_b = ss.csr_matrix(np.array([[0, 0, 0, 1], [1, 0, 0, 1], [0, 0, 0, 0]]))

    out1 = [[(0, 0, 1)], [(1, 1, 2)], None]
    assert list(top_cosine_similarities([X_a, X_b], chunksize=None, ntop=1)) == out1
    assert list(_cossim_top(X_a, X_b.T, ntop=1)) == out1

    out2 = [[(0, 1, 1), (0, 0, 1)], [(1, 1, 2), (1, 0, 1)], None]
    assert list(top_cosine_similarities([X_a, X_b], chunksize=None, ntop=2)) == out2
    assert list(_cossim_top(X_a, X_b.T, ntop=2)) == out2


def test_sparse_chunks():
    X_empty = ss.csr_matrix((4, 3))

    # Chunksize of one gives length of input
    assert len(_sparse_chunks(X_empty, 1)) == X_empty.shape[0]

    assert len(_sparse_chunks(X_empty, 2)) == 2
    assert len(_sparse_chunks(X_empty, 3)) == 2

    # Check overly large chunksize gives one chunks
    np.testing.assert_equal(_sparse_chunks(X_empty, 10)[0].todense(), X_empty.todense())
    assert len(_sparse_chunks(X_empty, 10)) == 1
    assert len(_sparse_chunks(X_empty, 4)) == 1

