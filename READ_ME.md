"Farkle Game Code.ipynb" is the main file to read about Farkle stats and the code generating it.  

TLDR: Going by the aggregate data, keep rolling if you have at least 3 dice and less than 350 points, or 300 if you are ahead and want to play extra-conservatively.  
If on the fence, score is a more important consideration than available dice.  

The top 50 strategies value was chosen arbitrarily but it turns out that they fall within the 96% confidence interval for significance of differing from the high count.
Put another way, if there is more than a 4% chance that a count is the same as the max count, 
that count and the top count get considered statistically the same, a statistic that applies to all counts among the top 50.
 
Somewhat shaky logic but it's enough of a gray area that I am fine sticking with top 50 rather than rewrite values and re-run the notebook, 
particularly because I had this realization months after the original data were made and analyzed.
Also, I've rolled too many natural 1s in DND to trust a P-value around 0.05 so I'd be comfortable saying more than 2% is too high a chance if I were to go back and use a statistically justified top n counts.

SWEET_VIZ Report Files are exploratory data analysis HTMLs, the no_interactions one is the most important one.


PNGs are screenshots of the EDA imported into the notebook with python because I've had too many issues with Jupyter Notebook image embedding in vscode.

CSVs are mostly data recording and transformation steps.

The important CSVs are:

- farkle_game_results_expanded.csv which lists the winner of all 80,000 games

- farkle_strategy_counts.csv which aggregates the winning strategies

- farkle_scores.csv is a list of all (I think, if not it is most) types of scoring rolls n dice can make.  
    The Scores, Used Dice, and Reroll Dice values are hand calculated and the csv was used for testing the dice roll evaluation logic.

- Smart_Comparison.csv which aggregates strategies where all other things kept the same, Smart == True outperformed Smart == False

- top_50_strategies.csv is a simple list of the top 50 strategies 

- top_50_strategies_weighted.csv is a list of the top 50 winning strategies where the strategy is listed as many times as it won in the 80,000 games.


There are many statistical comparisons and tests I could have done but the randomness is so apparent that I don't think it is worth the effort for something as low stakes as a dice game.