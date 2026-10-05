# Website-ready synthetic CSVs

Upload one of these if the website asks for the six Module A columns `delay_0h`, `delay_24h`, `leakage_0h`, `leakage_24h`, `lot_id`, and `serial_number`:

- `sih170_easy_demo_120_website_upload.csv`
- `sih170_balanced_120_website_upload.csv`
- `sih170_stress_120_website_upload.csv`

Each file contains 120 components, all 19 original generator columns, and six compatible aliases (25 columns total). The aliases are mapped as follows:

| Required column | Generator source |
| --- | --- |
| `serial_number` | `Component_ID` |
| `lot_id` | `Lot_ID` |
| `delay_0h` | `PropagationDelay_0h` |
| `delay_24h` | `PropagationDelay_24h` |
| `leakage_0h` | `Leakage_0h` |
| `leakage_24h` | `Leakage_24h` |

The aliases preserve the original values and units; delay remains in ns and leakage remains in uA. Other original columns are kept so you can compare forecasts with the synthetic 168-hour targets. All rows are marked `SYNTHETIC`; they are for upload-flow testing, not real chip data.
