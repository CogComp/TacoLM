## Pattern Extraction Samples
We provide a script `scripts/pattern_extraction.py` that takes SRL results and outputs formatted training data.

-  `data/samples/tmparg_collection_all.txt`: A samples SRL results, in the format of `sent \t prev_sent \t next_sent \t verb_pos \t tmparg_start_pos \t tmparg_end_pos \n` 

The original SRL parses on Gigaword is hundreds of GBs in size, if you want it, please contact the authors.

Then you can run `python scripts/pattern_extraction.py` to generate the formatted training file, which by default outputs to `data/samples/formatted_for_training.txt`
