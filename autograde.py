#!/usr/bin/env python3

import click
import os
import json
import re

@click.command()
@click.option("--submission", required=True,  type=click.Path(exists=True), help="Test directory")
@click.option("--results", required=True, type = click.File(mode="w"), help="Where to put results")
def autograde(submission=None, results=None):
    with open(os.path.join(submission, "test_gpt2.txt")) as f:
        feedback = ""
        text = f.read().strip()
        count = sum(1 for _ in re.finditer(r'\b%s\b' % re.escape("matmul_backward"), text))
        success = re.search("Trained\sby\s+\w+@ucr.edu", text) is not None 
        success += (count == 490)
        if re.search("Trained\sby\s+\w+@ucr.edu", text) is None:
            feedback += "Cannot find the e-mail address of the trainer;"
        if count != 490:
            feedback += "The count of function calls is not correct;"
        if re.search("matmul_backward", text) is None:
            feedback += "The most time-consuming function is not correct;"     

        # https://gradescope-autograders.readthedocs.io/en/latest/specs/#output-format
        json.dump(dict(output="The autograder ran.",
                       visibility="visible",
                       stdout_visibility="visible",
                       tests=[ dict(score=success*100/2,
                                    max_score=100,
                                    number="1",
                                    output=feedback,
                                    tags=[],
                                    visibility="visible",
                       )
                       ]#,
#                       leaderboard=[
 #                          dict(name="goodness", value=1 if success else 0)
                       #               ]
        ), results, sort_keys=True, indent=4)
        
if __name__== "__main__":
    autograde()
