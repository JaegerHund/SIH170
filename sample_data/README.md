# SIH170 website test CSVs

These CSVs were generated with the repository's `generate_synthetic_dataset()` function. They are test fixtures, not chip measurements.

| File | Profile | Rows | Lots | Seed |
| --- | --- | ---: | ---: | ---: |
| `sih170_easy_demo_120.csv` | `easy_demo` | 120 | 6 | 170 |
| `sih170_balanced_120.csv` | `balanced` | 120 | 6 | 170 |
| `sih170_stress_120.csv` | `stress` | 120 | 6 | 170 |

Each file has the same 19-column wide schema:

- IDs/context: `Component_ID`, `Lot_ID`, `Device_Type`, `Temperature`
- Synthetic annotations: `True_Behavior`, `Data_Source`, `Synthetic_Profile`
- Measurements: `Iddq`, `Leakage`, and `PropagationDelay`, each at `0h`, `24h`, `96h`, and `168h`

For a realistic forecast test, use only each parameter's `0h` and `24h` readings as prediction inputs. Keep `96h` and `168h` out of the prediction feature set; the `168h` column is the synthetic ground truth for checking forecast error afterward. `True_Behavior` is also a synthetic annotation, not a model input.

All three files use the same seed for repeatability, but the profile changes the behavior mix and generator settings. The behavior counts can be found by grouping the `True_Behavior` column.

## Website upload versions

The files in `website_uploads/` are copies with the same generated records plus the Module A field names shown by the website's CSV validator. Upload these if the site requires `delay_0h`, `delay_24h`, `leakage_0h`, `leakage_24h`, `lot_id`, and `serial_number`.

| Website field | Copied from generator field |
| --- | --- |
| `serial_number` | `Component_ID` |
| `lot_id` | `Lot_ID` |
| `delay_0h`, `delay_24h` | `PropagationDelay_0h`, `PropagationDelay_24h` |
| `leakage_0h`, `leakage_24h` | `Leakage_0h`, `Leakage_24h` |

The original generator columns are retained in the upload copies, including the synthetic 168-hour targets and labels. The extra alias columns contain the same values under the names expected by the site.
