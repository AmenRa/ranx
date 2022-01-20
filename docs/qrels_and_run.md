# Qrels and Run

The first step in the offline evaluation of the effectiveness of an IR system is the definition of a list of _query relevance judgments_ and the ranked lists of documents retrieved for those queries by the system.
To ease the managing of these data, `ranx` implements two dedicated Python classes: 1) [**Qrels**][ranx.Qrels] for the query relevance judgments and 2) [**Run**][ranx.Run] for the computed ranked lists.

::: ranx.qrels
::: ranx.run