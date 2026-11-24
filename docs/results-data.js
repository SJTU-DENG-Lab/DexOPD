// Quantitative benchmark, learning-dynamics, ablation, and sweep data from the DexOPD paper.
window.DEXOPD_RESULTS = {
  "table1": [
    {
      "name": "SD3.5-Medium",
      "group": "base",
      "geneval": 0.6446,
      "genevalAcc": 0.6336,
      "ocr": 0.5486,
      "aesthetic": 5.3899,
      "pickscore": 0.8614,
      "imagereward": 0.8383,
      "clipscore": 0.2895,
      "hps": 0.2924,
      "ours": false
    },
    {
      "name": "GenEval Teacher",
      "group": "specialists",
      "geneval": 0.9534,
      "genevalAcc": 0.9349,
      "ocr": 0.6515,
      "aesthetic": 5.2635,
      "pickscore": 0.8513,
      "imagereward": 0.9277,
      "clipscore": 0.2869,
      "hps": 0.2673,
      "ours": false
    },
    {
      "name": "OCR Teacher",
      "group": "specialists",
      "geneval": 0.6617,
      "genevalAcc": 0.6144,
      "ocr": 0.9435,
      "aesthetic": 5.2605,
      "pickscore": 0.858,
      "imagereward": 0.8972,
      "clipscore": 0.2909,
      "hps": 0.2732,
      "ours": false
    },
    {
      "name": "Aesthetic Teacher",
      "group": "specialists",
      "geneval": 0.4791,
      "genevalAcc": 0.7066,
      "ocr": 0.5881,
      "aesthetic": 6.2162,
      "pickscore": 0.9233,
      "imagereward": 1.5016,
      "clipscore": 0.296,
      "hps": 0.3452,
      "ours": false
    },
    {
      "name": "GRPO-Guard",
      "group": "rl",
      "geneval": 0.7202,
      "genevalAcc": 0.6764,
      "ocr": 0.721,
      "aesthetic": 5.6812,
      "pickscore": 0.877,
      "imagereward": 1.1531,
      "clipscore": 0.2904,
      "hps": 0.3091,
      "ours": false
    },
    {
      "name": "DiffusionNFT",
      "group": "rl",
      "geneval": 0.7029,
      "genevalAcc": 0.5817,
      "ocr": 0.9434,
      "aesthetic": 5.3674,
      "pickscore": 0.8412,
      "imagereward": 0.6314,
      "clipscore": 0.2787,
      "hps": 0.2642,
      "ours": false
    },
    {
      "name": "DiffusionART",
      "group": "rl",
      "geneval": 0.4427,
      "genevalAcc": 0.4847,
      "ocr": 0.9033,
      "aesthetic": 5.6673,
      "pickscore": 0.8292,
      "imagereward": 0.7193,
      "clipscore": 0.271,
      "hps": 0.2653,
      "ours": false
    },
    {
      "name": "DiffusionOPD",
      "group": "opd",
      "geneval": 0.9543,
      "genevalAcc": 0.9373,
      "ocr": 0.9371,
      "aesthetic": 6.182,
      "pickscore": 0.9237,
      "imagereward": 1.511,
      "clipscore": 0.3006,
      "hps": 0.3547,
      "ours": false
    },
    {
      "name": "DreOPD",
      "group": "opd",
      "geneval": 0.9622,
      "genevalAcc": 0.9495,
      "ocr": 0.9354,
      "aesthetic": 6.2182,
      "pickscore": 0.9237,
      "imagereward": 1.5259,
      "clipscore": 0.2986,
      "hps": 0.3558,
      "ours": false
    },
    {
      "name": "STEP-OPD*",
      "group": "opd",
      "geneval": 0.9562,
      "genevalAcc": 0.944,
      "ocr": 0.9378,
      "aesthetic": 6.2813,
      "pickscore": 0.9248,
      "imagereward": 1.5207,
      "clipscore": 0.2985,
      "hps": 0.3566,
      "ours": false
    },
    {
      "name": "DexOPD (ours)",
      "group": "opd",
      "geneval": 0.9736,
      "genevalAcc": 0.9566,
      "ocr": 0.9515,
      "aesthetic": 6.4875,
      "pickscore": 0.919,
      "imagereward": 1.5127,
      "clipscore": 0.2914,
      "hps": 0.3484,
      "ours": true
    }
  ],
  "ablation": [
    {
      "label": "DexOPD (default)",
      "desc": "Full method (\u03bb = 1.2, \u03b2 = 1.8, EMA H_r)",
      "isDefault": true,
      "geneval": 0.9736,
      "ocr": 0.9515,
      "aesthetic": 6.4875,
      "pickscore": 0.919,
      "imagereward": 1.5127,
      "clipscore": 0.2914,
      "hps": 0.3484
    },
    {
      "label": "No teacher extrap. (\u03bb = 0)",
      "desc": "Remove teacher extrapolation",
      "isDefault": false,
      "geneval": 0.9515,
      "ocr": 0.9357,
      "aesthetic": 6.293,
      "pickscore": 0.9215,
      "imagereward": 1.4903,
      "clipscore": 0.296,
      "hps": 0.3366
    },
    {
      "label": "No student aux. (\u03b2 = 0)",
      "desc": "Remove self-extrapolation reward branch",
      "isDefault": false,
      "geneval": 0.9669,
      "ocr": 0.9408,
      "aesthetic": 6.441,
      "pickscore": 0.9185,
      "imagereward": 1.4957,
      "clipscore": 0.2944,
      "hps": 0.3527
    },
    {
      "label": "Frozen-base ref. (H_r = B)",
      "desc": "Replace evolving EMA history with base model",
      "isDefault": false,
      "geneval": 0.9633,
      "ocr": 0.9512,
      "aesthetic": 6.2548,
      "pickscore": 0.8844,
      "imagereward": 1.4034,
      "clipscore": 0.2843,
      "hps": 0.3254
    }
  ],
  "betaSweep": [
    {
      "beta": 0.0,
      "geneval": 0.9669,
      "ocr": 0.9408,
      "aesthetic": 6.441,
      "pickscore": 0.9185,
      "imagereward": 1.4957,
      "clipscore": 0.2944,
      "hps": 0.3527
    },
    {
      "beta": 0.2,
      "geneval": 0.9715,
      "ocr": 0.9356,
      "aesthetic": 6.4511,
      "pickscore": 0.9185,
      "imagereward": 1.5086,
      "clipscore": 0.2922,
      "hps": 0.3485
    },
    {
      "beta": 0.4,
      "geneval": 0.9743,
      "ocr": 0.9414,
      "aesthetic": 6.4533,
      "pickscore": 0.919,
      "imagereward": 1.5036,
      "clipscore": 0.2911,
      "hps": 0.3478
    },
    {
      "beta": 0.6,
      "geneval": 0.9681,
      "ocr": 0.946,
      "aesthetic": 6.4629,
      "pickscore": 0.9187,
      "imagereward": 1.5023,
      "clipscore": 0.2912,
      "hps": 0.3484
    },
    {
      "beta": 0.8,
      "geneval": 0.9647,
      "ocr": 0.9452,
      "aesthetic": 6.4689,
      "pickscore": 0.9192,
      "imagereward": 1.5045,
      "clipscore": 0.2919,
      "hps": 0.3477
    },
    {
      "beta": 1.0,
      "geneval": 0.9698,
      "ocr": 0.9478,
      "aesthetic": 6.4594,
      "pickscore": 0.9195,
      "imagereward": 1.5124,
      "clipscore": 0.2952,
      "hps": 0.3532
    },
    {
      "beta": 1.2,
      "geneval": 0.9717,
      "ocr": 0.95,
      "aesthetic": 6.4718,
      "pickscore": 0.9198,
      "imagereward": 1.5123,
      "clipscore": 0.2919,
      "hps": 0.3481
    },
    {
      "beta": 1.4,
      "geneval": 0.9702,
      "ocr": 0.9458,
      "aesthetic": 6.4448,
      "pickscore": 0.9192,
      "imagereward": 1.5177,
      "clipscore": 0.2921,
      "hps": 0.3482
    },
    {
      "beta": 1.6,
      "geneval": 0.9526,
      "ocr": 0.9437,
      "aesthetic": 6.4622,
      "pickscore": 0.9201,
      "imagereward": 1.5118,
      "clipscore": 0.2919,
      "hps": 0.347
    },
    {
      "beta": 1.8,
      "geneval": 0.9736,
      "ocr": 0.9515,
      "aesthetic": 6.4875,
      "pickscore": 0.919,
      "imagereward": 1.5127,
      "clipscore": 0.2914,
      "hps": 0.3484
    },
    {
      "beta": 2.0,
      "geneval": 0.9587,
      "ocr": 0.952,
      "aesthetic": 6.4728,
      "pickscore": 0.9187,
      "imagereward": 1.5108,
      "clipscore": 0.292,
      "hps": 0.3471
    }
  ],
  "teacherSweeps": {
    "geneval": {
      "title": "GenEval Teacher",
      "onTaskKey": "geneval",
      "onTaskLabel": "GenEval reward",
      "offTaskKey": "imagereward",
      "offTaskLabel": "DrawBench ImageReward",
      "offTaskSecondaryKey": "ocr",
      "offTaskSecondaryLabel": "OCR score",
      "dexopdScore": 0.9736,
      "peakLambda": 1.4,
      "rows": [
        {
          "lambda": 0.0,
          "geneval": 0.9534,
          "ocr": 0.6515,
          "aesthetic": 5.2635,
          "pickscore": 0.8513,
          "imagereward": 0.9277,
          "clipscore": 0.2869,
          "hps": 0.2673
        },
        {
          "lambda": 0.2,
          "geneval": 0.9531,
          "ocr": 0.4157,
          "aesthetic": 5.2942,
          "pickscore": 0.8446,
          "imagereward": 0.5511,
          "clipscore": 0.2745,
          "hps": 0.2463
        },
        {
          "lambda": 0.4,
          "geneval": 0.9578,
          "ocr": 0.4302,
          "aesthetic": 5.2781,
          "pickscore": 0.8439,
          "imagereward": 0.5093,
          "clipscore": 0.2739,
          "hps": 0.2458
        },
        {
          "lambda": 0.6,
          "geneval": 0.9581,
          "ocr": 0.4316,
          "aesthetic": 5.2622,
          "pickscore": 0.8424,
          "imagereward": 0.4636,
          "clipscore": 0.2732,
          "hps": 0.2443
        },
        {
          "lambda": 0.8,
          "geneval": 0.9651,
          "ocr": 0.4559,
          "aesthetic": 5.2781,
          "pickscore": 0.842,
          "imagereward": 0.4716,
          "clipscore": 0.2717,
          "hps": 0.2452
        },
        {
          "lambda": 1.0,
          "geneval": 0.963,
          "ocr": 0.4775,
          "aesthetic": 5.2512,
          "pickscore": 0.841,
          "imagereward": 0.4545,
          "clipscore": 0.2709,
          "hps": 0.2439
        },
        {
          "lambda": 1.2,
          "geneval": 0.9652,
          "ocr": 0.4714,
          "aesthetic": 5.2611,
          "pickscore": 0.8401,
          "imagereward": 0.4643,
          "clipscore": 0.271,
          "hps": 0.2429
        },
        {
          "lambda": 1.4,
          "geneval": 0.9661,
          "ocr": 0.4985,
          "aesthetic": 5.2525,
          "pickscore": 0.8387,
          "imagereward": 0.4002,
          "clipscore": 0.2681,
          "hps": 0.2418
        },
        {
          "lambda": 1.6,
          "geneval": 0.9637,
          "ocr": 0.4733,
          "aesthetic": 5.2501,
          "pickscore": 0.8375,
          "imagereward": 0.4041,
          "clipscore": 0.2672,
          "hps": 0.2415
        },
        {
          "lambda": 1.8,
          "geneval": 0.9597,
          "ocr": 0.4818,
          "aesthetic": 5.2543,
          "pickscore": 0.8364,
          "imagereward": 0.3683,
          "clipscore": 0.2648,
          "hps": 0.24
        },
        {
          "lambda": 2.0,
          "geneval": 0.9623,
          "ocr": 0.4872,
          "aesthetic": 5.2526,
          "pickscore": 0.8358,
          "imagereward": 0.3536,
          "clipscore": 0.2645,
          "hps": 0.2387
        }
      ]
    },
    "ocr": {
      "title": "OCR Teacher",
      "onTaskKey": "ocr",
      "onTaskLabel": "OCR score",
      "offTaskKey": "imagereward",
      "offTaskLabel": "DrawBench ImageReward",
      "offTaskSecondaryKey": "geneval",
      "offTaskSecondaryLabel": "GenEval reward",
      "dexopdScore": 0.9515,
      "peakLambda": 0.8,
      "rows": [
        {
          "lambda": 0.0,
          "geneval": 0.6617,
          "ocr": 0.9435,
          "aesthetic": 5.2605,
          "pickscore": 0.858,
          "imagereward": 0.8972,
          "clipscore": 0.2909,
          "hps": 0.2732
        },
        {
          "lambda": 0.2,
          "geneval": 0.6214,
          "ocr": 0.9566,
          "aesthetic": 5.2438,
          "pickscore": 0.8391,
          "imagereward": 0.6339,
          "clipscore": 0.2912,
          "hps": 0.2508
        },
        {
          "lambda": 0.4,
          "geneval": 0.5914,
          "ocr": 0.958,
          "aesthetic": 5.1923,
          "pickscore": 0.8357,
          "imagereward": 0.5848,
          "clipscore": 0.2891,
          "hps": 0.2456
        },
        {
          "lambda": 0.6,
          "geneval": 0.5765,
          "ocr": 0.9523,
          "aesthetic": 5.1718,
          "pickscore": 0.8314,
          "imagereward": 0.5108,
          "clipscore": 0.2879,
          "hps": 0.2423
        },
        {
          "lambda": 0.8,
          "geneval": 0.5613,
          "ocr": 0.9616,
          "aesthetic": 5.1091,
          "pickscore": 0.8262,
          "imagereward": 0.4202,
          "clipscore": 0.2855,
          "hps": 0.237
        },
        {
          "lambda": 1.0,
          "geneval": 0.5574,
          "ocr": 0.9534,
          "aesthetic": 5.0735,
          "pickscore": 0.8223,
          "imagereward": 0.3461,
          "clipscore": 0.2816,
          "hps": 0.2333
        },
        {
          "lambda": 1.2,
          "geneval": 0.5145,
          "ocr": 0.9559,
          "aesthetic": 5.0281,
          "pickscore": 0.8166,
          "imagereward": 0.2917,
          "clipscore": 0.2807,
          "hps": 0.2299
        },
        {
          "lambda": 1.4,
          "geneval": 0.5004,
          "ocr": 0.9544,
          "aesthetic": 4.9383,
          "pickscore": 0.8101,
          "imagereward": 0.0857,
          "clipscore": 0.2765,
          "hps": 0.222
        },
        {
          "lambda": 1.6,
          "geneval": 0.4695,
          "ocr": 0.9541,
          "aesthetic": 4.8835,
          "pickscore": 0.8055,
          "imagereward": -0.0227,
          "clipscore": 0.2711,
          "hps": 0.2177
        },
        {
          "lambda": 1.8,
          "geneval": 0.4459,
          "ocr": 0.9552,
          "aesthetic": 4.8332,
          "pickscore": 0.7994,
          "imagereward": -0.1318,
          "clipscore": 0.2665,
          "hps": 0.2133
        },
        {
          "lambda": 2.0,
          "geneval": 0.4276,
          "ocr": 0.9487,
          "aesthetic": 4.7608,
          "pickscore": 0.7959,
          "imagereward": -0.2367,
          "clipscore": 0.2646,
          "hps": 0.2095
        }
      ]
    },
    "aesthetic": {
      "title": "Aesthetic Teacher",
      "onTaskKey": "aesthetic",
      "onTaskLabel": "DrawBench Aesthetic",
      "offTaskKey": "geneval",
      "offTaskLabel": "GenEval reward",
      "offTaskSecondaryKey": "ocr",
      "offTaskSecondaryLabel": "OCR score",
      "dexopdScore": 6.4875,
      "peakLambda": 0.8,
      "rows": [
        {
          "lambda": 0.0,
          "geneval": 0.4791,
          "ocr": 0.5881,
          "aesthetic": 6.2162,
          "pickscore": 0.9233,
          "imagereward": 1.5016,
          "clipscore": 0.296,
          "hps": 0.3452
        },
        {
          "lambda": 0.2,
          "geneval": 0.5288,
          "ocr": 0.6173,
          "aesthetic": 6.3069,
          "pickscore": 0.9222,
          "imagereward": 1.4916,
          "clipscore": 0.2962,
          "hps": 0.3425
        },
        {
          "lambda": 0.4,
          "geneval": 0.5011,
          "ocr": 0.6017,
          "aesthetic": 6.3491,
          "pickscore": 0.9228,
          "imagereward": 1.5071,
          "clipscore": 0.2959,
          "hps": 0.3444
        },
        {
          "lambda": 0.6,
          "geneval": 0.4711,
          "ocr": 0.5949,
          "aesthetic": 6.3883,
          "pickscore": 0.9204,
          "imagereward": 1.5077,
          "clipscore": 0.2941,
          "hps": 0.3446
        },
        {
          "lambda": 0.8,
          "geneval": 0.4499,
          "ocr": 0.5686,
          "aesthetic": 6.4164,
          "pickscore": 0.917,
          "imagereward": 1.4932,
          "clipscore": 0.294,
          "hps": 0.344
        },
        {
          "lambda": 1.0,
          "geneval": 0.4418,
          "ocr": 0.5491,
          "aesthetic": 6.4108,
          "pickscore": 0.9128,
          "imagereward": 1.4785,
          "clipscore": 0.2925,
          "hps": 0.3427
        },
        {
          "lambda": 1.2,
          "geneval": 0.3903,
          "ocr": 0.5269,
          "aesthetic": 6.4047,
          "pickscore": 0.9068,
          "imagereward": 1.4589,
          "clipscore": 0.2928,
          "hps": 0.3398
        },
        {
          "lambda": 1.4,
          "geneval": 0.3926,
          "ocr": 0.5126,
          "aesthetic": 6.3586,
          "pickscore": 0.9007,
          "imagereward": 1.4189,
          "clipscore": 0.2917,
          "hps": 0.3364
        },
        {
          "lambda": 1.6,
          "geneval": 0.3331,
          "ocr": 0.4981,
          "aesthetic": 6.276,
          "pickscore": 0.8935,
          "imagereward": 1.3907,
          "clipscore": 0.2924,
          "hps": 0.3307
        },
        {
          "lambda": 1.8,
          "geneval": 0.2851,
          "ocr": 0.5207,
          "aesthetic": 6.1857,
          "pickscore": 0.8847,
          "imagereward": 1.3369,
          "clipscore": 0.2908,
          "hps": 0.324
        },
        {
          "lambda": 2.0,
          "geneval": 0.2288,
          "ocr": 0.4858,
          "aesthetic": 6.0806,
          "pickscore": 0.8759,
          "imagereward": 1.2604,
          "clipscore": 0.2891,
          "hps": 0.316
        }
      ]
    }
  },
  "learningDynamics": {
    "steps": [
      0,
      100,
      200,
      300,
      400,
      500,
      600,
      700,
      800,
      900,
      1000
    ],
    "genevalOPD": {
      "rlTeacher": 0.9534,
      "proxyTeacher": 0.9652,
      "series": [
        {
          "id": "dexopd",
          "label": "DexOPD (ours)",
          "values": [
            0.6446,
            0.9006,
            0.9393,
            0.9504,
            0.9612,
            0.9596,
            0.9651,
            0.9607,
            0.9667,
            0.9697,
            0.9736
          ]
        },
        {
          "id": "dreopd",
          "label": "DreOPD",
          "values": [
            0.6446,
            0.8276,
            0.9022,
            0.9356,
            0.9452,
            0.9586,
            0.9557,
            0.958,
            0.9589,
            0.9597,
            0.9622
          ]
        },
        {
          "id": "diffusionopd",
          "label": "DiffusionOPD",
          "values": [
            0.6446,
            0.8759,
            0.9386,
            0.9477,
            0.9563,
            0.9502,
            0.9534,
            0.9577,
            0.9598,
            0.9573,
            0.9543
          ]
        }
      ]
    },
    "multitask": {
      "dexopd": {
        "label": "DexOPD (ours)",
        "summary": "Steady progress across all three tasks over 1,000 updates.",
        "geneval": [
          0.6446,
          0.9006,
          0.9393,
          0.9504,
          0.9612,
          0.9596,
          0.9651,
          0.9607,
          0.9667,
          0.9697,
          0.9736
        ],
        "ocr": [
          0.5486,
          0.941,
          0.9414,
          0.9469,
          0.9394,
          0.9439,
          0.9478,
          0.9347,
          0.9499,
          0.944,
          0.9515
        ],
        "aesthetic": [
          5.3899,
          5.9619,
          6.3685,
          6.3422,
          6.3893,
          6.3873,
          6.4236,
          6.4479,
          6.474,
          6.4557,
          6.4875
        ]
      },
      "nft": {
        "label": "Multi-task DiffusionNFT",
        "summary": "OCR climbs, while GenEval fluctuates and Aesthetic regresses below base.",
        "geneval": [
          0.6446,
          0.6224,
          0.599,
          0.6688,
          0.7099,
          0.6677,
          0.749,
          0.6701,
          0.6548,
          0.719,
          0.7029
        ],
        "ocr": [
          0.5486,
          0.7467,
          0.8334,
          0.8813,
          0.901,
          0.8966,
          0.9367,
          0.9205,
          0.9228,
          0.9135,
          0.9434
        ],
        "aesthetic": [
          5.3899,
          5.528,
          5.7271,
          5.7177,
          5.9072,
          5.9597,
          5.7754,
          5.4742,
          5.7107,
          5.6308,
          5.3674
        ]
      },
      "art": {
        "label": "Multi-task DiffusionART",
        "summary": "Mid-training GenEval collapses to 0.4427 despite rising OCR.",
        "geneval": [
          0.6446,
          0.5816,
          0.6412,
          0.5693,
          0.6212,
          0.6909,
          0.5059,
          0.3974,
          0.4428,
          0.4457,
          0.4427
        ],
        "ocr": [
          0.5486,
          0.7017,
          0.8526,
          0.8622,
          0.8657,
          0.8745,
          0.883,
          0.8696,
          0.879,
          0.8629,
          0.9033
        ],
        "aesthetic": [
          5.3899,
          5.2719,
          5.2166,
          5.6419,
          5.685,
          5.5793,
          5.5136,
          5.1594,
          5.3289,
          5.6825,
          5.6673
        ]
      }
    }
  },
  "efficiency": [
    {
      "method": "DiffusionOPD",
      "batch": 72,
      "geneval": 0.9543,
      "ocr": 0.9371,
      "aesthetic": 6.182,
      "sec": 25.76,
      "ours": false
    },
    {
      "method": "DreOPD",
      "batch": 24,
      "geneval": 0.9622,
      "ocr": 0.9354,
      "aesthetic": 6.2182,
      "sec": 9.73,
      "ours": false
    },
    {
      "method": "STEP-OPD*",
      "batch": 72,
      "geneval": 0.9562,
      "ocr": 0.9378,
      "aesthetic": 6.2813,
      "sec": 32.53,
      "ours": false
    },
    {
      "method": "DexOPD (small batch)",
      "batch": 24,
      "geneval": 0.968,
      "ocr": 0.9488,
      "aesthetic": 6.4857,
      "sec": 25.59,
      "ours": true,
      "highlight": "fast"
    },
    {
      "method": "DexOPD (default)",
      "batch": 72,
      "geneval": 0.9736,
      "ocr": 0.9515,
      "aesthetic": 6.4875,
      "sec": 68.18,
      "ours": true,
      "highlight": "default"
    }
  ]
};
