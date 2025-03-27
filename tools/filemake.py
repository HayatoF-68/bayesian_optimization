import os

# 出力ファイルのパスを指定
output_file_path = '~/Documents/Projects/bayesian_optimization-master/bayesian_optimization/output/20240706023130/combined_scores.txt'

# 読み込むファイルのディレクトリパスを指定
input_directory = '~/Documents/Projects/bayesian_optimization-master/bayesian_optimization/output/20240706023130/'

# 読み込むファイル名のフォーマットを指定
file_pattern = 'epoch{0}/score.txt'

# 0から100までのscore.txtを読み込み、combined_scores.txtに書き出す
with open(os.path.expanduser(output_file_path), 'w') as output_file:
    for epoch in range(100):
        input_file_path = os.path.join(input_directory, file_pattern.format(epoch))
        with open(os.path.expanduser(input_file_path), 'r') as input_file:
            score = input_file.read().strip()
            output_file.write(f'{score}\n')
