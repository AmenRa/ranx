# Fusion

## Fuse
---
`ranx` provides several [fusion algorithms][supported-fusion-algorithms], all of which can be accessed through a single function in the same fashion as `evaluate`.

Usage example:
```python
from ranx import fuse

combined_run = fuse(
    runs=[run_1, run_2],  # A list of Run instances to fuse
    norm="min-max",       # The normalization strategy to apply before fusion
    method="max",         # The fusion algorithm to use 
)
```

## Optimize Fusion
---
As many fusion algorithms require a training or optimization step, `ranx` provides
a function to optimize all of those algorithms. For algorithms requiring
hyper-parameter optimization, `ranx` automatically evaluates pre-defined
configurations via grid search. In those cases, `ranx` shows a progress bar.

Usage example:
```python
from ranx import fuse, optimize_fusion

best_params = optimize_fusion(
    qrels=train_qrels,
    runs=[train_run_1, train_run_2, train_run_3],
    norm="min-max",
    method="wsum",
    metric="ndcg@100",  # The metric to maximize during optimization
)

combined_test_run = fuse(
    runs=[test_run_1, test_run_2, test_run_3],  
    norm="min-max",       
    method="wsum",        
    params=best_params,
)
```

## Supported fusion algorithms
---

`ranx` supports the following fusion algorithms:

| **Algorithm**                                              | **Alias** | **Optim.** | **Algorithm**                            | **Alias**   | **Optim.** |
| ---------------------------------------------------------- | --------- | :--------: | ---------------------------------------- | ----------- | :--------: |
| [CombMIN][combmin]                                         | min       |     No     | [CombMAX][combmax]                       | max         |     No     |
| [CombMED][combmed]                                         | med       |     No     | [CombSUM][combsum]                       | sum         |     No     |
| [CombANZ][combanz]                                         | anz       |     No     | [CombMNZ][combmnz]                       | mnz         |     No     |
| [CombGMNZ][combgmnz]                                       | gmnz      |    Yes     | [ISR][isr]                               | isr         |     No     |
| [Log_ISR][log_isr]                                         | log_isr   |     No     | [LogN_ISR][logn_isr]                     | logn_isr    |    Yes     |
| [Reciprocal Rank Fusion (RRF)][reciprocal-rank-fusion-rrf] | rrf       |    Yes     | [PosFuse][posfuse]                       | posfuse     |    Yes     |
| [ProbFuse][probfuse]                                       | probfuse  |    Yes     | [SegFuse][segfuse]                       | segfuse     |    Yes     |
| [SlideFuse][slidefuse]                                     | slidefuse |    Yes     | [MAPFuse][mapfuse]                       | mapfuse     |    Yes     |
| [BordaFuse][bordafuse]                                     | bordafuse |     No     | [Weighted BordaFuse][weighted-bordafuse] | w_bordafuse |    Yes     |
| [Condorcet][condorcet]                                     | condorcet |     No     | [Weighted Condorcet][weighted-condorcet] | w_condorcet |    Yes     |
| [BayesFuse][bayesfuse]                                     | bayesfuse |    Yes     | [Mixed][mixed]                           | mixed       |    Yes     |
| [WMNZ][wmnz]                                               | wmnz      |    Yes     | [Wighted Sum][wighted-sum]               | wsum        |    Yes     |
| [Rank-Biased Centroids (RBC)][rank-biased-centroids-rbc]   | rbc       |    Yes     |                                          |             |


### BayesFuse
Computes BayesFuse as proposed by [Aslam et al.](https://dl.acm.org/doi/10.1145/383952.384007).  
<details>
    <summary>BibTeX</summary>
    ```bibtex
    @inproceedings{DBLP:conf/sigir/AslamM01,
        author    = {Javed A. Aslam and
                    Mark H. Montague},
        editor    = {W. Bruce Croft and
                    David J. Harper and
                    Donald H. Kraft and
                    Justin Zobel},
        title     = {Models for Metasearch},
        booktitle = {{SIGIR} 2001: Proceedings of the 24th Annual International {ACM} {SIGIR}
                    Conference on Research and Development in Information Retrieval, September
                    9-13, 2001, New Orleans, Louisiana, {USA}},
        pages     = {275--284},
        publisher = {{ACM}},
        year      = {2001},
        url       = {https://doi.org/10.1145/383952.384007},
        doi       = {10.1145/383952.384007},
        timestamp = {Tue, 06 Nov 2018 11:07:25 +0100},
        biburl    = {https://dblp.org/rec/conf/sigir/AslamM01.bib},
        bibsource = {dblp computer science bibliography, https://dblp.org}
    }
    ```
</details>


### BordaFuse
Computes BordaFuse as proposed by [Aslam et al.](https://dl.acm.org/doi/10.1145/383952.384007).  
<details>
    <summary>BibTeX</summary>
    ```bibtex
    @inproceedings{DBLP:conf/sigir/AslamM01,
        author    = {Javed A. Aslam and
                    Mark H. Montague},
        editor    = {W. Bruce Croft and
                    David J. Harper and
                    Donald H. Kraft and
                    Justin Zobel},
        title     = {Models for Metasearch},
        booktitle = {{SIGIR} 2001: Proceedings of the 24th Annual International {ACM} {SIGIR}
                    Conference on Research and Development in Information Retrieval, September
                    9-13, 2001, New Orleans, Louisiana, {USA}},
        pages     = {275--284},
        publisher = {{ACM}},
        year      = {2001},
        url       = {https://doi.org/10.1145/383952.384007},
        doi       = {10.1145/383952.384007},
        timestamp = {Tue, 06 Nov 2018 11:07:25 +0100},
        biburl    = {https://dblp.org/rec/conf/sigir/AslamM01.bib},
        bibsource = {dblp computer science bibliography, https://dblp.org}
    }
    ```
</details>


### CombANZ
Computes CombANZ as proposed by [Fox et al.](https://trec.nist.gov/pubs/trec2/papers/txt/23.txt).  
<details>
    <summary>BibTeX</summary>
    ```bibtex
    @inproceedings{DBLP:conf/trec/FoxS93,
        author    = {Edward A. Fox and
                    Joseph A. Shaw},
        title     = {Combination of Multiple Searches},
        booktitle = {{TREC}},
        series    = {{NIST} Special Publication},
        volume    = {500-215},
        pages     = {243--252},
        publisher = {National Institute of Standards and Technology {(NIST)}},
        year      = {1993}
    }
    ```
</details>


### CombGMNZ
Computes CombGMNZ as proposed by [Joon Ho Lee](https://dl.acm.org/doi/10.1145/258525.258587).
<details>
    <summary>BibTeX</summary>
    ```bibtex
    @inproceedings{DBLP:conf/sigir/Lee97,
        author    = {Joon Ho Lee},
        title     = {Analyses of Multiple Evidence Combination},
        booktitle = {{SIGIR}},
        pages     = {267--276},
        publisher = {{ACM}},
        year      = {1997}
    }
    ```
</details>
| **Optimization Parameter** | **Default Value** |
| -------------------------- | :---------------: |
| min_gamma                  |       0.01        |
| max_gamma                  |        1.0        |
| step                       |       0.01        |

### CombMAX
Computes CombMAX as proposed by [Fox et al.](https://trec.nist.gov/pubs/trec2/papers/txt/23.txt).
<details>
    <summary>BibTeX</summary>
    ```bibtex
    @inproceedings{DBLP:conf/trec/FoxS93,
        author    = {Edward A. Fox and
                    Joseph A. Shaw},
        title     = {Combination of Multiple Searches},
        booktitle = {{TREC}},
        series    = {{NIST} Special Publication},
        volume    = {500-215},
        pages     = {243--252},
        publisher = {National Institute of Standards and Technology {(NIST)}},
        year      = {1993}
    }
    ```
</details>


### CombMED
Computes CombMED as proposed by [Fox et al.](https://trec.nist.gov/pubs/trec2/papers/txt/23.txt).
<details>
    <summary>BibTeX</summary>
    ```bibtex
    @inproceedings{DBLP:conf/trec/FoxS93,
        author    = {Edward A. Fox and
                    Joseph A. Shaw},
        title     = {Combination of Multiple Searches},
        booktitle = {{TREC}},
        series    = {{NIST} Special Publication},
        volume    = {500-215},
        pages     = {243--252},
        publisher = {National Institute of Standards and Technology {(NIST)}},
        year      = {1993}
    }
    ```
</details>


### CombMIN
Computes CombMIN as proposed by [Fox et al.](https://trec.nist.gov/pubs/trec2/papers/txt/23.txt).
<details>
    <summary>BibTeX</summary>
    ```bibtex
    @inproceedings{DBLP:conf/trec/FoxS93,
        author    = {Edward A. Fox and
                    Joseph A. Shaw},
        title     = {Combination of Multiple Searches},
        booktitle = {{TREC}},
        series    = {{NIST} Special Publication},
        volume    = {500-215},
        pages     = {243--252},
        publisher = {National Institute of Standards and Technology {(NIST)}},
        year      = {1993}
    }
    ```
</details>


### CombMNZ
Computes CombMNZ as proposed by [Fox et al.](https://trec.nist.gov/pubs/trec2/papers/txt/23.txt).
<details>
    <summary>BibTeX</summary>
    ```bibtex
    @inproceedings{DBLP:conf/trec/FoxS93,
        author    = {Edward A. Fox and
                    Joseph A. Shaw},
        title     = {Combination of Multiple Searches},
        booktitle = {{TREC}},
        series    = {{NIST} Special Publication},
        volume    = {500-215},
        pages     = {243--252},
        publisher = {National Institute of Standards and Technology {(NIST)}},
        year      = {1993}
    }
    ```
</details>


### CombSUM
Computes CombSUM as proposed by [Fox et al.](https://trec.nist.gov/pubs/trec2/papers/txt/23.txt).
<details>
    <summary>BibTeX</summary>
    ```bibtex
    @inproceedings{DBLP:conf/trec/FoxS93,
        author    = {Edward A. Fox and
                    Joseph A. Shaw},
        title     = {Combination of Multiple Searches},
        booktitle = {{TREC}},
        series    = {{NIST} Special Publication},
        volume    = {500-215},
        pages     = {243--252},
        publisher = {National Institute of Standards and Technology {(NIST)}},
        year      = {1993}
    }
    ```
</details>


### Condorcet
Computes Condorcet as proposed by [Montague et al.](https://dl.acm.org/doi/10.1145/584792.584881).
<details>
    <summary>BibTeX</summary>
    ```bibtex
    @inproceedings{DBLP:conf/cikm/MontagueA02,
        author    = {Mark H. Montague and
                    Javed A. Aslam},
        title     = {Condorcet fusion for improved retrieval},
        booktitle = {Proceedings of the 2002 {ACM} {CIKM} International Conference on Information
                    and Knowledge Management, McLean, VA, USA, November 4-9, 2002},
        pages     = {538--548},
        publisher = {{ACM}},
        year      = {2002},
        url       = {https://doi.org/10.1145/584792.584881},
        doi       = {10.1145/584792.584881},
        timestamp = {Tue, 06 Nov 2018 16:57:50 +0100},
        biburl    = {https://dblp.org/rec/conf/cikm/MontagueA02.bib},
        bibsource = {dblp computer science bibliography, https://dblp.org}
    }
    ```
</details>

### ISR
Computes ISR as proposed by [Mourão et al.](https://www.sciencedirect.com/science/article/abs/pii/S0895611114000664).
<details>
    <summary>BibTeX</summary>
    ```bibtex
    @article{DBLP:journals/cmig/MouraoMM15,
        author    = {Andr{\'{e}} Mour{\~{a}}o and
                    Fl{\'{a}}vio Martins and
                    Jo{\~{a}}o Magalh{\~{a}}es},
        title     = {Multimodal medical information retrieval with unsupervised rank fusion},
        journal   = {Comput. Medical Imaging Graph.},
        volume    = {39},
        pages     = {35--45},
        year      = {2015},
        url       = {https://doi.org/10.1016/j.compmedimag.2014.05.006},
        doi       = {10.1016/j.compmedimag.2014.05.006},
        timestamp = {Thu, 14 May 2020 10:17:16 +0200},
        biburl    = {https://dblp.org/rec/journals/cmig/MouraoMM15.bib},
        bibsource = {dblp computer science bibliography, https://dblp.org}
    }
    ```
</details>


### Log_ISR
Computes Log_ISR as proposed by [Mourão et al.](https://www.sciencedirect.com/science/article/abs/pii/S0895611114000664).
<details>
    <summary>BibTeX</summary>
    ```bibtex
    @article{DBLP:journals/cmig/MouraoMM15,
        author    = {Andr{\'{e}} Mour{\~{a}}o and
                    Fl{\'{a}}vio Martins and
                    Jo{\~{a}}o Magalh{\~{a}}es},
        title     = {Multimodal medical information retrieval with unsupervised rank fusion},
        journal   = {Comput. Medical Imaging Graph.},
        volume    = {39},
        pages     = {35--45},
        year      = {2015},
        url       = {https://doi.org/10.1016/j.compmedimag.2014.05.006},
        doi       = {10.1016/j.compmedimag.2014.05.006},
        timestamp = {Thu, 14 May 2020 10:17:16 +0200},
        biburl    = {https://dblp.org/rec/journals/cmig/MouraoMM15.bib},
        bibsource = {dblp computer science bibliography, https://dblp.org}
    }
    ```
</details>


### LogN_ISR
Computes Log_ISR as proposed by [Mourão et al.](https://www.sciencedirect.com/science/article/abs/pii/S0895611114000664).
<details>
    <summary>BibTeX</summary>
    ```bibtex
    @article{DBLP:journals/cmig/MouraoMM15,
        author    = {Andr{\'{e}} Mour{\~{a}}o and
                    Fl{\'{a}}vio Martins and
                    Jo{\~{a}}o Magalh{\~{a}}es},
        title     = {Multimodal medical information retrieval with unsupervised rank fusion},
        journal   = {Comput. Medical Imaging Graph.},
        volume    = {39},
        pages     = {35--45},
        year      = {2015},
        url       = {https://doi.org/10.1016/j.compmedimag.2014.05.006},
        doi       = {10.1016/j.compmedimag.2014.05.006},
        timestamp = {Thu, 14 May 2020 10:17:16 +0200},
        biburl    = {https://dblp.org/rec/journals/cmig/MouraoMM15.bib},
        bibsource = {dblp computer science bibliography, https://dblp.org}
    }
    ```
</details>
| **Optimization Parameter** | **Default Value** |
| -------------------------- | :---------------: |
| min_sigma                  |       0.01        |
| max_sigma                  |        1.0        |
| step                       |       0.01        |


### MAPFuse
Computes MAPFuse as proposed by [Lillis et al.](https://dl.acm.org/doi/10.1145/1835449.1835508).
<details>
    <summary>BibTeX</summary>
    ```bibtex
    @inproceedings{DBLP:conf/sigir/LillisZTCLD10,
        author    = {David Lillis and
                    Lusheng Zhang and
                    Fergus Toolan and
                    Rem W. Collier and
                    David Leonard and
                    John Dunnion},
        editor    = {Fabio Crestani and
                    St{\'{e}}phane Marchand{-}Maillet and
                    Hsin{-}Hsi Chen and
                    Efthimis N. Efthimiadis and
                    Jacques Savoy},
        title     = {Estimating probabilities for effective data fusion},
        booktitle = {Proceeding of the 33rd International {ACM} {SIGIR} Conference on Research
                    and Development in Information Retrieval, {SIGIR} 2010, Geneva, Switzerland,
                    July 19-23, 2010},
        pages     = {347--354},
        publisher = {{ACM}},
        year      = {2010},
        url       = {https://doi.org/10.1145/1835449.1835508},
        doi       = {10.1145/1835449.1835508},
        timestamp = {Tue, 06 Nov 2018 11:07:25 +0100},
        biburl    = {https://dblp.org/rec/conf/sigir/LillisZTCLD10.bib},
        bibsource = {dblp computer science bibliography, https://dblp.org}
    }
    ```
</details>


### Mixed
Computes Mixed as proposed by [Wu et al.](https://dl.acm.org/doi/10.1145/584792.584908).
<details>
    <summary>BibTeX</summary>
    ```bibtex
    @inproceedings{DBLP:conf/cikm/WuC02,
        author    = {Shengli Wu and
                    Fabio Crestani},
        title     = {Data fusion with estimated weights},
        booktitle = {Proceedings of the 2002 {ACM} {CIKM} International Conference on Information
                    and Knowledge Management, McLean, VA, USA, November 4-9, 2002},
        pages     = {648--651},
        publisher = {{ACM}},
        year      = {2002},
        url       = {https://doi.org/10.1145/584792.584908},
        doi       = {10.1145/584792.584908},
        timestamp = {Tue, 06 Nov 2018 16:57:40 +0100},
        biburl    = {https://dblp.org/rec/conf/cikm/WuC02.bib},
        bibsource = {dblp computer science bibliography, https://dblp.org}
    }
    ```
</details>
| **Optimization Parameter** | **Default Value** |
| -------------------------- | :---------------: |
| step                       |        0.1        |


### PosFuse
Computes PosFuse as proposed by [Lillis et al.](https://dl.acm.org/doi/10.1145/1835449.1835508).
<details>
    <summary>BibTeX</summary>
    ```bibtex
    @inproceedings{DBLP:conf/sigir/LillisZTCLD10,
        author    = {David Lillis and
                    Lusheng Zhang and
                    Fergus Toolan and
                    Rem W. Collier and
                    David Leonard and
                    John Dunnion},
        editor    = {Fabio Crestani and
                    St{\'{e}}phane Marchand{-}Maillet and
                    Hsin{-}Hsi Chen and
                    Efthimis N. Efthimiadis and
                    Jacques Savoy},
        title     = {Estimating probabilities for effective data fusion},
        booktitle = {Proceeding of the 33rd International {ACM} {SIGIR} Conference on Research
                    and Development in Information Retrieval, {SIGIR} 2010, Geneva, Switzerland,
                    July 19-23, 2010},
        pages     = {347--354},
        publisher = {{ACM}},
        year      = {2010},
        url       = {https://doi.org/10.1145/1835449.1835508},
        doi       = {10.1145/1835449.1835508},
        timestamp = {Tue, 06 Nov 2018 11:07:25 +0100},
        biburl    = {https://dblp.org/rec/conf/sigir/LillisZTCLD10.bib},
        bibsource = {dblp computer science bibliography, https://dblp.org}
    }
    ```
</details>


### ProbFuse
Computes ProbFuse as proposed by [Lillis et al.](https://dl.acm.org/doi/10.1145/1148170.1148197).
<details>
    <summary>BibTeX</summary>
    ```bibtex
    @inproceedings{DBLP:conf/sigir/LillisTCD06,
        author    = {David Lillis and
                    Fergus Toolan and
                    Rem W. Collier and
                    John Dunnion},
        editor    = {Efthimis N. Efthimiadis and
                    Susan T. Dumais and
                    David Hawking and
                    Kalervo J{\"{a}}rvelin},
        title     = {ProbFuse: a probabilistic approach to data fusion},
        booktitle = {{SIGIR} 2006: Proceedings of the 29th Annual International {ACM} {SIGIR}
                    Conference on Research and Development in Information Retrieval, Seattle,
                    Washington, USA, August 6-11, 2006},
        pages     = {139--146},
        publisher = {{ACM}},
        year      = {2006},
        url       = {https://doi.org/10.1145/1148170.1148197},
        doi       = {10.1145/1148170.1148197},
        timestamp = {Wed, 14 Nov 2018 10:58:10 +0100},
        biburl    = {https://dblp.org/rec/conf/sigir/LillisTCD06.bib},
        bibsource = {dblp computer science bibliography, https://dblp.org}
    }
    ```
</details>
| **Optimization Parameter** | **Default Value** |
| -------------------------- | :---------------: |
| min_n_segments             |         1         |
| max_n_segments             |        100        |


### Rank-Biased Centroids (RBC)
Computes Rank-Biased Centroid (RBC) as proposed by [Bailey et al.](https://dl.acm.org/doi/10.1145/3077136.3080839).
<details>
    <summary>BibTeX</summary>
    ```bibtex
    @inproceedings{DBLP:conf/sigir/BaileyMST17,
        author    = {Peter Bailey and
                    Alistair Moffat and
                    Falk Scholer and
                    Paul Thomas},
        editor    = {Noriko Kando and
                    Tetsuya Sakai and
                    Hideo Joho and
                    Hang Li and
                    Arjen P. de Vries and
                    Ryen W. White},
        title     = {Retrieval Consistency in the Presence of Query Variations},
        booktitle = {Proceedings of the 40th International {ACM} {SIGIR} Conference on
                    Research and Development in Information Retrieval, Shinjuku, Tokyo,
                    Japan, August 7-11, 2017},
        pages     = {395--404},
        publisher = {{ACM}},
        year      = {2017},
        url       = {https://doi.org/10.1145/3077136.3080839},
        doi       = {10.1145/3077136.3080839},
        timestamp = {Wed, 25 Sep 2019 16:43:14 +0200},
        biburl    = {https://dblp.org/rec/conf/sigir/BaileyMST17.bib},
        bibsource = {dblp computer science bibliography, https://dblp.org}
    }
    ```
</details>
| **Optimization Parameter** | **Default Value** |
| -------------------------- | :---------------: |
| min_phi                    |       0.01        |
| max_phi                    |        1.0        |
| step                       |       0.01        |

### Reciprocal Rank Fusion (RRF)
Computes Reciprocal Rank Fusion as proposed by [Cormack et al.](https://dl.acm.org/doi/10.1145/1571941.1572114).
<details>
    <summary>BibTeX</summary>
    ```bibtex
    @inproceedings{DBLP:conf/sigir/CormackCB09,
        author    = {Gordon V. Cormack and
                    Charles L. A. Clarke and
                    Stefan B{\"{u}}ttcher},
        title     = {Reciprocal rank fusion outperforms condorcet and individual rank learning
                    methods},
        booktitle = {{SIGIR}},
        pages     = {758--759},
        publisher = {{ACM}},
        year      = {2009}
    }
    ```
</details>
| **Optimization Parameter** | **Default Value** |
| -------------------------- | :---------------: |
| min_k                      |        10         |
| max_k                      |        100        |
| step                       |        10         |


### SegFuse
Computes SegFuse as proposed by [Shokouhi](https://link.springer.com/chapter/10.1007/978-3-540-78646-7_33).
<details>
    <summary>BibTeX</summary>
    ```bibtex
    @inproceedings{DBLP:conf/ecir/Shokouhi07a,
        author    = {Milad Shokouhi},
        editor    = {Giambattista Amati and
                    Claudio Carpineto and
                    Giovanni Romano},
        title     = {Segmentation of Search Engine Results for Effective Data-Fusion},
        booktitle = {Advances in Information Retrieval, 29th European Conference on {IR}
                    Research, {ECIR} 2007, Rome, Italy, April 2-5, 2007, Proceedings},
        series    = {Lecture Notes in Computer Science},
        volume    = {4425},
        pages     = {185--197},
        publisher = {Springer},
        year      = {2007},
        url       = {https://doi.org/10.1007/978-3-540-71496-5\_19},
        doi       = {10.1007/978-3-540-71496-5\_19},
        timestamp = {Tue, 14 May 2019 10:00:37 +0200},
        biburl    = {https://dblp.org/rec/conf/ecir/Shokouhi07a.bib},
        bibsource = {dblp computer science bibliography, https://dblp.org}
    }
    ```
</details>


### SlideFuse
Computes SlideFuse as proposed by [Lillis et al.](https://link.springer.com/chapter/10.1007/978-3-540-78646-7_33).
<details>
    <summary>BibTeX</summary>
    ```bibtex
    @inproceedings{DBLP:conf/ecir/LillisTCD08,
        author    = {David Lillis and
                    Fergus Toolan and
                    Rem W. Collier and
                    John Dunnion},
        editor    = {Craig Macdonald and
                    Iadh Ounis and
                    Vassilis Plachouras and
                    Ian Ruthven and
                    Ryen W. White},
        title     = {Extending Probabilistic Data Fusion Using Sliding Windows},
        booktitle = {Advances in Information Retrieval , 30th European Conference on {IR}
                    Research, {ECIR} 2008, Glasgow, UK, March 30-April 3, 2008. Proceedings},
        series    = {Lecture Notes in Computer Science},
        volume    = {4956},
        pages     = {358--369},
        publisher = {Springer},
        year      = {2008},
        url       = {https://doi.org/10.1007/978-3-540-78646-7\_33},
        doi       = {10.1007/978-3-540-78646-7\_33},
        timestamp = {Sun, 25 Oct 2020 22:33:08 +0100},
        biburl    = {https://dblp.org/rec/conf/ecir/LillisTCD08.bib},
        bibsource = {dblp computer science bibliography, https://dblp.org}
    }
    ```
</details>
| **Optimization Parameter** | **Default Value** |
| -------------------------- | :---------------: |
| min_w                      |         1         |
| max_w                      |        100        |


### Weighted BordaFuse
Computes Weighted BordaFuse as proposed by [Aslam et al.](https://dl.acm.org/doi/10.1145/383952.384007).
<details>
    <summary>BibTeX</summary>
    ```bibtex
    @inproceedings{DBLP:conf/sigir/AslamM01,
        author    = {Javed A. Aslam and
                    Mark H. Montague},
        editor    = {W. Bruce Croft and
                    David J. Harper and
                    Donald H. Kraft and
                    Justin Zobel},
        title     = {Models for Metasearch},
        booktitle = {{SIGIR} 2001: Proceedings of the 24th Annual International {ACM} {SIGIR}
                    Conference on Research and Development in Information Retrieval, September
                    9-13, 2001, New Orleans, Louisiana, {USA}},
        pages     = {275--284},
        publisher = {{ACM}},
        year      = {2001},
        url       = {https://doi.org/10.1145/383952.384007},
        doi       = {10.1145/383952.384007},
        timestamp = {Tue, 06 Nov 2018 11:07:25 +0100},
        biburl    = {https://dblp.org/rec/conf/sigir/AslamM01.bib},
        bibsource = {dblp computer science bibliography, https://dblp.org}
    }
    ```
</details>
| **Optimization Parameter** | **Default Value** |
| -------------------------- | :---------------: |
| step                       |        0.1        |


### Weighted Condorcet
Computes Weighted Condorcet as proposed by [Montague et al.](https://dl.acm.org/doi/10.1145/584792.584881).
<details>
    <summary>BibTeX</summary>
    ```bibtex
    @inproceedings{DBLP:conf/cikm/MontagueA02,
        author    = {Mark H. Montague and
                    Javed A. Aslam},
        title     = {Condorcet fusion for improved retrieval},
        booktitle = {Proceedings of the 2002 {ACM} {CIKM} International Conference on Information
                    and Knowledge Management, McLean, VA, USA, November 4-9, 2002},
        pages     = {538--548},
        publisher = {{ACM}},
        year      = {2002},
        url       = {https://doi.org/10.1145/584792.584881},
        doi       = {10.1145/584792.584881},
        timestamp = {Tue, 06 Nov 2018 16:57:50 +0100},
        biburl    = {https://dblp.org/rec/conf/cikm/MontagueA02.bib},
        bibsource = {dblp computer science bibliography, https://dblp.org}
    }
    ```
</details>
| **Optimization Parameter** | **Default Value** |
| -------------------------- | :---------------: |
| step                       |        0.1        |

### WMNZ
Computes Weighted MNZ as proposed by [Wu et al.](https://dl.acm.org/doi/10.1145/584792.584908).
<details>
    <summary>BibTeX</summary>
    ```bibtex
    @inproceedings{DBLP:conf/cikm/WuC02,
        author    = {Shengli Wu and
                    Fabio Crestani},
        title     = {Data fusion with estimated weights},
        booktitle = {Proceedings of the 2002 {ACM} {CIKM} International Conference on Information
                    and Knowledge Management, McLean, VA, USA, November 4-9, 2002},
        pages     = {648--651},
        publisher = {{ACM}},
        year      = {2002},
        url       = {https://doi.org/10.1145/584792.584908},
        doi       = {10.1145/584792.584908},
        timestamp = {Tue, 06 Nov 2018 16:57:40 +0100},
        biburl    = {https://dblp.org/rec/conf/cikm/WuC02.bib},
        bibsource = {dblp computer science bibliography, https://dblp.org}
    }
    ```
</details>
| **Optimization Parameter** | **Default Value** |
| -------------------------- | :---------------: |
| step                       |        0.1        |


### Wighted Sum
Computes a weighted sum of the scores given to documents by a list of Runs.


| **Optimization Parameter** | **Default Value** |
| -------------------------- | :---------------: |
| step                       |        0.1        |

