import os
import pickle
import sys
import argparse
import multiprocessing
def parse_model_output_file(dist_label):
	total_result = []
	version_list = []
	for group_index in range(1, 11):
		with open("./result/{}/{}.txt".format(dist_label,group_index), "r") as file:
			content = file.readlines()
		current_result = []
		first_flag = True
		for line in content:
			line = line.strip()
			if line.startswith("==="):
				if len(version_list) > 0:
					if len(current_result) == 0:
						if not first_flag:
							version_list.pop()
					else:
						total_result.append(current_result[:])
						current_result = []
				version_list.append(line.replace("===", ""))
			else:
				current_result.append(float(line))
			first_flag = False
		if len(current_result) > 0:
			total_result.append(current_result[:])
		else:
			version_list.pop()
	return total_result, version_list


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='train')

	parser.add_argument('--batch', type=int, default=64)
	parser.add_argument('--epoch', type=int, default=60)
	parser.add_argument('--model',type=str,default='Conpre')
	parser.add_argument('--dataset',type=str,default='squeeze_40_20_new')
	parser.add_argument('--seed', type=int, default=81402)
	parser.add_argument('--hidden-dim', type=int, default=50)
	parser.add_argument('--layer-num', type=int, default=1)
	parser.add_argument('--loss',type=str,default='hinge')
	parser.add_argument('--lr', type=float, default=0.01)
	parser.add_argument('--weight-decay', type=float, default=1e-4)
	parser.add_argument('--vector-file', type=str,  default='vectors_64')
	parser.add_argument('--cnn-channel', type=int, default=2)
	parser.add_argument("--restore-mode",type=int,default=0)

	args = parser.parse_args()
	for group_index in range(1,11):
		status = os.system("python test.py --task {} --batch {} --epoch {} --model {} --dataset {} --seed {} --hidden-dim {} --layer-num {} --loss {} --lr {} --weight-decay {} --vector-file {} --cnn-channel {}"
		.format(group_index,args.batch,args.epoch,args.model,args.dataset,args.seed,args.hidden_dim,args.layer_num,args.loss,args.lr,args.weight_decay,args.vector_file,args.cnn_channel))
		assert status == 0

	top1_total = 0
	top3_total = 0
	top5_total = 0
	all_position_total = []
	first_position_total = []
	distinct_label="{}_{}_{}_{}_{}_{}_{}".format(args.model,args.dataset,args.batch,args.epoch,args.layer_num,str(args.lr).replace('.','-'),str(args.cnn_channel))
	total_result, version_list = parse_model_output_file(distinct_label)
	ff=open("./result/total_{}.txt".format(distinct_label),"w",encoding="utf8")
	print("\nStatistics for each project.")

	for project in ["Chart", "Closure", "Math", "Mockito", "Lang", "Time"]:

		top1 = 0
		top3 = 0
		top5 = 0
		all_position = []
		first_position = []

		for i, version in enumerate(version_list):
			if not version.startswith(project):
				continue
			bugs = total_result[i]
			rank = []
			for bug in bugs:
				rank.append(bug)
			min_rank = min(rank)
			avg_rank = sum(rank) / len(rank)

			if min_rank <= 1:
				top1 += 1
			if min_rank <= 3:
				top3 += 1
			if min_rank <= 5:
				top5 += 1
			for current_rank in rank:
				if min_rank==current_rank:
					first_position.append(min_rank)
			all_position.append(avg_rank)

		ff.write("=" * 20)
		ff.write("\n")
		ff.write(project)
		ff.write("\n")
		ff.write("Top1\t{}\n".format(top1))
		ff.write("Top3\t{}\n".format(top3))
		ff.write("Top5\t{}\n".format(top5))
		ff.write("MFR\t{}\n".format(round(sum(first_position) / len(first_position), 2)))
		ff.write("MAR\t{}\n".format(round(sum(all_position) / len(all_position), 2)))

		top1_total += top1
		top3_total += top3
		top5_total += top5
		all_position_total += all_position
		first_position_total += first_position

	ff.write("\nStatistics for all projects.\n")
	ff.write("=" * 20)
	ff.write("\n")
	ff.write("Top1\t{}\n".format(top1_total))
	ff.write("Top3\t{}\n".format(top3_total))
	ff.write("Top5\t{}\n".format(top5_total))
	ff.write("MFR\t{}\n".format(round(sum(first_position_total) / len(first_position_total), 2)))
	ff.write("MAR\t{}\n".format(round(sum(all_position_total) / len(all_position_total), 2)))
	ff.close()

