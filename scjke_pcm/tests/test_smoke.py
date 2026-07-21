import numpy as np
import pandas as pd

from scjke_pcm import AdaptiveSCJKEPCM


def source_frame():
    return pd.DataFrame(
        {
            "SMILES": [
                "CCO", "CCCO", "CC(=O)O", "CCC(=O)O",
                "c1ccccc1", "Cc1ccccc1", "CCN", "CCCN",
            ],
            "Protein": [
                "MKTAYIAKQRQISFVKSHFSRQDILDLWQ",
                "MKTAYIAKQRQISFVKSHFSRQDILDLWE",
                "TTCCPSIVARSNFNVCRLPGTPEAICAT",
                "TTCCPSIVARSNFNVCRLPGTPEAICAA",
                "GAVLILKKKGHHEAELKPLAQSHATKHK",
                "GAVLILKKKGHHEAELKPLAQSHATKHA",
                "MALWMRLLPLLALLALWGPDPAAA",
                "MALWMRLLPLLALLALWGPDPVVV",
            ],
            "Y": [1, 1, 1, 1, 0, 0, 0, 0],
        }
    )


def query_frame():
    return pd.DataFrame(
        {
            "SMILES": ["CCCO", "c1ccccc1O"],
            "Protein": [
                "MKTAYIAKQRQISFVKSHFSRQDILDLWA",
                "GAVLILKKKGHHEAELKPLAQSHATKHT",
            ],
            "Y": [1, 0],
        }
    )


def test_predictions_do_not_use_query_labels():
    model = AdaptiveSCJKEPCM(top_k=2, fingerprint_bits=256).fit(source_frame())
    query = query_frame()
    first = model.predict_dataframe(query)["p_dti"].to_numpy()
    query["Y"] = 1 - query["Y"]
    second = model.predict_dataframe(query)["p_dti"].to_numpy()
    np.testing.assert_allclose(first, second)
    assert np.all((first >= 0.0) & (first <= 1.0))


def test_source_only_weight_is_bounded():
    model = AdaptiveSCJKEPCM(top_k=2, fingerprint_bits=256).fit(source_frame())
    assert 0.10 <= model.pcm_weight <= 0.30
