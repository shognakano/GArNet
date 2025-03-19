import os,sys,re,random,shutil,copy,time
from pyrosetta import*
from pyrosetta.toolbox import*
from pyrosetta.rosetta import*
from pyrosetta.rosetta.core.scoring import*
from pyrosetta.rosetta.protocols.simple_moves import*
from pyrosetta.rosetta.protocols.minimization_packing import*
from Bio import AlignIO
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

start_time_1 = time.time()
init()

# ********************Phase.I:[Extract mutational condidates (using GAOptimizer) ]********************

class STANDARD_TOOLS():
# Commonly used tools.

	def PDB_TO_SEQ(input_pdb,chain_name):
	# Extract residue, residue number and pose from pdb.

		picked_residue = [];picked_residue_number = []
		input_pose = Pose()
		input_pose = pose_from_pdb(input_pdb)
		residue_number = input_pose.total_residue()		
		for a in range(residue_number):
			tmp_chain = input_pose.pdb_info().chain(a+1)
			tmp_residue = input_pose.residue(a+1).name1()
			if re.search(chain_name,tmp_chain):
				tmp_picked_residue_number = str(a+1)
				picked_residue.append(tmp_residue)
				picked_residue_number.append(tmp_picked_residue_number)
			else:
				continue
									
		return picked_residue,picked_residue_number,input_pose
	
	def NULL_ELIMINATE(csv_list):
	# Eliminate null data.

		after_data = []
		for b in range(len(csv_list)):
			tmp = csv_list[b]
			if re.search(".*\w+.*",tmp):
				after_data.append(tmp)
			else:
				continue
		
		return after_data

class MAFFTtoINTMSAlign():
# Calculate the amino acid frequency of each residue number from the MAFFT.

	def __init__(self,library_directory,stp_residue):

		self.library_file_data = os.listdir(library_directory)
		with open(library_directory+"/"+self.library_file_data[np.random.randint(0,len(self.library_file_data))], "r") as fh_1:
			self.open_library_data = fh_1.read()

		self.stp_seq = "".join(stp_residue)
		self.stp_tag = "template"
		self.mafft_input = f">{self.stp_tag}\n{self.stp_seq}\n"
		self.mafft_input += self.open_library_data
		with open("tmp_mafft.inp", "w") as fh_2:
			fh_2.write(self.mafft_input)
		self.mafft_output = "tmp_mafft.out"

		os.system(f"mafft tmp_mafft.inp > {self.mafft_output}")

		self.all_labels,self.all_sequences = self.CLASSIFICATION(self.mafft_output)
		
		self.stp_tag_check = "no"
		for d in range(len(self.all_labels)):
			self.tmp_label_data = self.all_labels[d]
			if re.search(self.stp_tag,self.tmp_label_data):
				self.stp_index = int(self.all_labels.index(self.tmp_label_data))
				self.stp_tag_check = "yes"
		if self.stp_tag_check == "no":
			print("INPUT ERROR : Please input the STP name correctively !")
			sys.exit(1)
		
		self.amino_rates_matrix,self.stp_residues = self.AMINO_COUNTER(self.all_sequences,self.stp_index)
		
		self.intmsa_output_data = "    Residue: [ ALA   CYS   ASP   GLU   PHE   GLY   HIS   ILE   LYS   LEU   MET   ASN   PRO   GLN   ARG   SER   THR   VAL   TRP   TYR   Non ]\n"
		self.column_number = int(0)
		for i in range(len(self.amino_rates_matrix)):
			self.column_number += 1
			preserve_number = int(0)
			stp_residue = self.stp_residues[i]
			amino_rates = self.amino_rates_matrix[i]
			self.intmsa_output_data += f"{str(self.column_number).ljust(4)} {preserve_number}   {stp_residue} : "
			for j in range(len(amino_rates)):
				amino_rate = f"{amino_rates[j]:.1f}"
				self.intmsa_output_data += f"{str(amino_rate).rjust(5)} "
			self.intmsa_output_data += "\n"
		
		with open("tmp_intmsa.out", "w") as fh_3:
			fh_3.write(self.intmsa_output_data)

	def CLASSIFICATION(self,mafft_data):
	# Classify MAFFT results into labels and sequences.
		
		all_labels = [];all_sequences = []
		alignment = AlignIO.read(open(mafft_data), "fasta")
		for c in range(len(alignment)):
			tmp_label = "".join(alignment[c].id)
			tmp_sequence = "".join(alignment[c].seq)
			all_labels.append(tmp_label)
			all_sequences.append(tmp_sequence)
		
		return all_labels,all_sequences

	def AMINO_COUNTER(self,all_sequences,stp_index):
	# Count the number of amino acids.
	
		residue_index_list = [];stp_residues = []
		stp_sequence = all_sequences[stp_index]
		for e in range(len(stp_sequence)):
			tmp_stp_letter = stp_sequence[e]
			if re.search("-",tmp_stp_letter):
				continue
			else:
				residue_index_list.append(e)
				stp_residues.append(tmp_stp_letter)
		
		amino_counter_matrix = np.zeros((len(residue_index_list),21),int)
		amino_rates_matrix = np.zeros((len(residue_index_list),21),float)
		amino_21_letter = ["A","C","D","E","F","G","H","I","K","L","M","N","P","Q","R","S","T","V","W","Y","-"]
		for f in range(len(residue_index_list)):
			tmp_index = int(residue_index_list[f])
			for g in range(len(all_sequences)):
				tmp_residue = all_sequences[g][tmp_index]
				pass_flag = int(0)
				number = int(0)
				while (pass_flag == int(0)) and (number != 21):
					if re.search(amino_21_letter[number],tmp_residue):
						amino_counter_matrix[f][number] += 1
						pass_flag = int(1)
					else:
						number += 1
		
		for h in range(len(amino_counter_matrix)):
			amino_rates_matrix[h] = amino_counter_matrix[h]/np.sum(amino_counter_matrix[h])*100.0
		
		return amino_rates_matrix,stp_residues

class CALCULATION_SCORE():
# Minimize energy and calculate Rosetta Energy Units (REU) and HiSol Score.<FITNESS_FUNCTION>
	
	def __init__(self,pose,task_pack_mut,now_generation_number,selective_pressure,initial_flag,intmsa_data,stp_pose,library_hisol_score):# <FITNESS_FUNCTION>
		
		self.pose = pose
		self.task_pack_mut = task_pack_mut
		self.scorefxn = ScoreFunction()
		self.scorefxn = get_fa_scorefxn()

		self.refinement = self.ENERGY_MINIMIZE(self.pose,self.task_pack_mut,self.scorefxn)

		self.refine = Pose()
		self.refine.assign(self.pose)
		self.refinement.apply(self.refine)
		self.score_refine = self.scorefxn(self.refine)
		self.score_init = self.scorefxn(self.pose)
				
		self.reu_score = self.score_refine# <FITNESS_FUNCTION>-(START)

		if initial_flag == int(0):
			if selective_pressure == "HISOL":
				self.library_hisol_score = library_hisol_score

				self.hisol_score,self.library_hisol_score = self.HISOL_SCORE(now_generation_number,self.pose,intmsa_data,stp_pose,self.library_hisol_score)# <FITNESS_FUNCTION>-(GOAL)

	def ENERGY_MINIMIZE(self,pose,task_pack_mut,scorefxn):
	# Minimize energy.
		
		kT = 1.0
		cycles = 1
		movemap = MoveMap()
		movemap.set_bb(False)
		minmover = MinMover()
		minmover.movemap(movemap)
		minmover.score_function(scorefxn)
		combined_mover = SequenceMover()
		combined_mover.add_mover(minmover)
		
		print(task_pack_mut)
		if len(task_pack_mut) > 0:
			task_pack = standard_packer_task(pose)
			task_pack.restrict_to_repacking()
			task_pack.temporarily_fix_everything()
			for p in range(len(task_pack_mut)):
				tmp_mut = task_pack_mut[p]
				task_pack.temporarily_set_pack_residue(tmp_mut,True)
			taskpackmover = PackRotamersMover(scorefxn,task_pack)
			combined_mover.add_mover(taskpackmover)
		
		mc = MonteCarlo(pose,scorefxn,kT)
		trial = TrialMover(combined_mover,mc)
		refinement = RepeatMover(trial,cycles)
		
		return refinement
	
	def HISOL_LIBRARY(self,intmsa_data,hisol_residue_index):
	# Calculate hydrophobicity for each residue number from the amino acid frequency of the "library" sequence.
				
		library_all_residues_frequency_matrix = np.zeros((len(intmsa_data),20),float)
		for q in range(len(intmsa_data)):
			tmp_residue_number_data = intmsa_data[q].split(":")[1]
			tmp_residue_number_data_2 = re.split("\s+",tmp_residue_number_data)
			tmp_residue_number_data_3 = STANDARD_TOOLS.NULL_ELIMINATE(tmp_residue_number_data_2)			
			for r in range(20):
				tmp_frequency_value = float(0)
				tmp_frequency_value = tmp_residue_number_data_3[r]
				library_all_residues_frequency_matrix[q][r] = tmp_frequency_value

		amino_acid_one_letter = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']
		final_hisol_score_from_library = np.zeros(len(intmsa_data),float)
		for s in range(len(intmsa_data)):
			tmp_residue_number_frequency = library_all_residues_frequency_matrix[s]
			tmp_sum_frequency = float(0)
			for t in range(len(tmp_residue_number_frequency)):
				tmp_sum_frequency += tmp_residue_number_frequency[t]/float(100)*float(hisol_residue_index[str(amino_acid_one_letter[t])])
			final_hisol_score_from_library[s] = tmp_sum_frequency
		
		return final_hisol_score_from_library

	def HISOL_SCORE(self,now_generation_number,pose,intmsa_data,stp_pose,library_hisol_score):
	# Calculate the HiSol Score from the difference in hydrophobicity of each residue number in the target and library sequences.

		hisol_residue_index = {'I':'4.5','V':'4.2','L':'3.8','F':'2.8','C':'2.5','M':'1.9','A':'1.8','G':'-0.4','T':'-0.7','S':'-0.8','W':'-0.9','Y':'-1.3','P':'-1.6','H':'-3.2','E':'-3.5','Q':'-3.5','D':'-3.5','N':'-3.5','K':'-3.9','R':'-4.5'}
		
		if now_generation_number == int(0):
			library_hisol_score = float(0)

			library_hisol_score = self.HISOL_LIBRARY(intmsa_data,hisol_residue_index)
			
		difference_score = float(0)
		mutation_sequence = pose.chain_sequence(1)
		mutation_hisol_score = np.zeros(len(mutation_sequence),float)
		for ee in range(len(mutation_sequence)):
			tmp_value = float(0)
			tmp_amino_residue = str(mutation_sequence[ee])
			tmp_value = float(hisol_residue_index[tmp_amino_residue])
			mutation_hisol_score[ee] = tmp_value
			
		difference_score = float(np.sum(np.abs(mutation_hisol_score-library_hisol_score)))
				
		scorefxn2 = ScoreFunction()
		scorefxn2 = get_fa_scorefxn()
		if scorefxn2(pose)-scorefxn2(stp_pose) > float(0):
			difference_score = float(100000)
			print("Destabilized !!!")

		return difference_score,library_hisol_score

class MUTATION():
# Mutagenesis creates many child-generations and selects the next parent-generations and elite by score.

	def __init__(self,random_mutation_percent,now_generation_number,recombination_mutation_percent_increasing,sample_number,random_mutation_incidence,parent_candidate_number,directory_name,chain_name,parent_sequence,parent_residue_number,intmsa_data,parent_pose,task_pack_mut,selective_pressure,initial_flag,stp_pose,library_hisol_score,parent_pdb,X_mutations_list_X,frequency_border_line,tournament_entry_number,percent_determine_generation):# <FITNESS_FUNCTION>
		
		if now_generation_number >= int(0):
		# During Phase.I.
			
			# _______________(Select each mutagenic rate.)_______________

			self.random_mutation_percent = random_mutation_percent
			self.recombination_mutation_percent = int(100)-int(self.random_mutation_percent)
			self.parent_pose = parent_pose
			self.task_pack_mut = task_pack_mut
			self.parent_pdb = parent_pdb
		
			if now_generation_number < percent_determine_generation:
				tmp_recombination_mutation_percent = now_generation_number*int(recombination_mutation_percent_increasing)
				if tmp_recombination_mutation_percent >= int(100)-int(self.random_mutation_percent):
					self.recombination_mutation_percent = int(100)-int(self.random_mutation_percent)
				else:
					self.recombination_mutation_percent = int(tmp_recombination_mutation_percent)
			else:
				self.recombination_mutation_percent = int(100)-int(self.random_mutation_percent)

			# _______________(Create the child-generation samples.)_______________
			
			self.parent_sequence = parent_sequence
			self.parent_residue_number = parent_residue_number
			self.output_pdbs_list = []
			self.reu_scores_array = np.zeros(sample_number,float)# <FITNESS_FUNCTION>-(START)
			if selective_pressure == "HISOL":
				self.hisol_scores_array = np.zeros(sample_number,float)# <FITNESS_FUNCTION>-(GOAL)
	
			self.mutations_directory_list = os.listdir(f"{directory_name}/temporary_mutations")
			if len(self.mutations_directory_list) > int(0):
				os.system(f"mv {directory_name}/temporary_mutations/* {directory_name}/temporary_previous_mutations/")
			
			self.mutations_parent_file_index = int(1000)
			
			for v in range(sample_number):

				# _____(Mutagenesis.)_____
			
				self.random_number = int(np.random.randint(1,int(sample_number+1),1))
				if now_generation_number == int(0):
					random_mutation_flag = int(1)
				elif self.random_number-self.recombination_mutation_percent > int(0):
					random_mutation_flag = int(1)
				else:
					random_mutation_flag = int(0)
			
				if now_generation_number >= int(1):
					self.next_parent_number = np.random.randint(0,parent_candidate_number)
					self.next_parent_pdb_list = os.listdir(f"{directory_name}/temporary_selected")
					self.next_parent_pdb_name = self.next_parent_pdb_list[self.next_parent_number]
					self.parent_pdb = f"{directory_name}/temporary_selected/{self.next_parent_pdb_name}"
					self.parent_sequence,self.parent_residue_number,self.parent_pose = STANDARD_TOOLS.PDB_TO_SEQ(self.parent_pdb,chain_name)
					self.parent_pdb = self.parent_pdb.split('/')[-1]
					self.mutations_parent_file_index = self.next_parent_number
	
				if random_mutation_flag == int(1):
				
					self.random_mutations = self.RANDOM(self.parent_sequence,self.parent_residue_number,random_mutation_incidence,intmsa_data,frequency_border_line)
	
					self.mutations = self.random_mutations
					print(f"{v+1} {self.mutations}")

				elif random_mutation_flag == int(0):

					self.recombination_mutations,self.template_pose,self.parent_pdb,self.mutations_parent_file_index = self.RECOMBINATION(directory_name,chain_name)

					self.mutations = self.recombination_mutations
					self.parent_pose = self.template_pose
					self.parent_pdb = self.parent_pdb.split('/')[-1]
					print(f"{v+1} {self.mutations}")
				
				self.mutations_output_data = []
				for ggg in range(len(self.mutations)):
					self.mutations_output_data.append(self.mutations[ggg].replace(":",""))			
				self.mutations_output = f"{now_generation_number}:{self.mutations_output_data}\n"

				tmp_file_handle = []
				if self.mutations_parent_file_index == int(1000):
					tmp_file_handle.append(f"fh_{now_generation_number}-{v}_w")
					with open(f"{directory_name}/temporary_sample_mutations/{v}.fasta","w") as tmp_file_handle[-1]:
						tmp_file_handle[-1].write(self.mutations_output)
					tmp_file_handle = []
				else:
					tmp_file_handle.append(f"fh_{now_generation_number}-{v}_a")
					self.mutations_parent_file_list = os.listdir(f"{directory_name}/temporary_previous_mutations")
					self.mutations_parent_file = self.mutations_parent_file_list[self.mutations_parent_file_index]
					os.system(f"cp {directory_name}/temporary_previous_mutations/{self.mutations_parent_file} {directory_name}/temporary_sample_mutations/{v}.fasta")
					with open(f"{directory_name}/temporary_sample_mutations/{v}.fasta","a") as tmp_file_handle[-1]:
						tmp_file_handle[-1].write(self.mutations_output)
					tmp_file_handle = []

				self.mutation_pose,self.task_pack_mut = self.MUTATE_POSE(self.parent_pose,self.mutations,self.task_pack_mut)

				# _____(Calculate scores.)_____
					
				calculation_score = CALCULATION_SCORE(self.mutation_pose,self.task_pack_mut,now_generation_number,selective_pressure,initial_flag,intmsa_data,stp_pose,library_hisol_score)# <FITNESS_FUNCTION>

				if selective_pressure == "HISOL":# <FITNESS_FUNCTION>-(START)
					self.library_hisol_score = calculation_score.library_hisol_score
				else:
					self.library_hisol_score = int(0)# <FITNESS_FUNCTION>-(GOAL)

				self.mutation_reu_score = calculation_score.reu_score# <FITNESS_FUNCTION>-(START)
				if selective_pressure == "HISOL":
					self.mutation_hisol_score = calculation_score.hisol_score# <FITNESS_FUNCTION>-(GOAL)
				self.mutation_pose.dump_pdb(f"{directory_name}/temporary_pyrosetta/{v}.pdb")	
				self.reu_scores_array[v] = self.mutation_reu_score# <FITNESS_FUNCTION>-(START)
				if selective_pressure == "HISOL":
					self.hisol_scores_array[v] = self.mutation_hisol_score# <FITNESS_FUNCTION>-(GOAL)
				self.output_pdbs_list.append(f"{v}.pdb")
	
			if selective_pressure == "REU":# <FITNESS_FUNCTION>-(START)
				self.evaluation_scores_array = self.reu_scores_array
			elif selective_pressure == "HISOL":
				self.evaluation_scores_array = self.hisol_scores_array# <FITNESS_FUNCTION>-(GOAL)
			else:
				print("INPUT ERROR : What is fitness function ?")
				sys.exit(1)
			
			# _____(Save score elite PDB.)_____
			
			self.min_score = np.min(self.evaluation_scores_array)
			self.min_score_index = np.argmin(self.evaluation_scores_array)
			self.elite_pose = self.output_pdbs_list[self.min_score_index]
			self.sample_mutations_list = os.listdir(f"{directory_name}/temporary_sample_mutations")
			self.elite_mutations_file = self.sample_mutations_list[self.min_score_index]
			self.elite_pdb_name = f"{str(now_generation_number)}-elite.pdb"
			os.system(f"cp {directory_name}/temporary_pyrosetta/{self.elite_pose} {directory_name}/temporary_elite/{self.elite_pdb_name}")
			
			# _____(Select the next parent-generation samples.)_____

			os.system(f"rm -f {directory_name}/temporary_selected/*")
			os.system(f"rm -f {directory_name}/temporary_mutations/*")
	
			self.min_scores_to_next_parent = np.zeros(int(parent_candidate_number)-1,float)
			for ff in range(int(parent_candidate_number)-1):

				self.winner_pdb_data,self.winner_min_score = self.TOURNAMENT(self.output_pdbs_list,self.evaluation_scores_array,directory_name,ff,tournament_entry_number)

				self.min_scores_to_next_parent[ff] = self.winner_min_score
				os.system(f"cp {directory_name}/temporary_pyrosetta/{self.winner_pdb_data} {directory_name}/temporary_selected/{ff+1}.pdb")
			os.system(f"cp {directory_name}/temporary_elite/{self.elite_pdb_name} {directory_name}/temporary_selected/0.pdb")
			os.system(f"cp {directory_name}/temporary_sample_mutations/{self.elite_mutations_file} {directory_name}/temporary_mutations/0.fasta")

			self.selected_average_score = np.average(self.min_scores_to_next_parent)# Except for elite.			
			self.scorefxn2 = ScoreFunction()
			self.scorefxn2 = get_fa_scorefxn()
			self.elite_pose_data = pose_from_pdb(f"{directory_name}/temporary_elite/{self.elite_pdb_name}")
			self.elite_rosetta_score = self.scorefxn2(self.elite_pose_data)
				
			self.return_elite_min_score = self.min_score
			self.return_selected_average_score = self.selected_average_score
			self.return_elite_energy_score = self.elite_rosetta_score

		else:
		# Final mutagenesis in Phase.II.
		
			self.mutations_list_final = X_mutations_list_X

			self.tmp_mutation_pose,self.task_pack_mut = self.MUTATE_POSE(parent_pose,self.mutations_list_final,task_pack_mut)

	def SELECT_RANDOM_SITE(self,mutation_residue_number,parent_residue_number,intmsa_data,frequency_border_line):
	# Determine consensus residues and select a residue after mutagenesis.

		frequency_array = np.zeros(20,float)
		consensus_residue_indexes = []

		residue_number_intmsa_data = intmsa_data[mutation_residue_number]
		frequency_data = residue_number_intmsa_data.split(":")[1]
		frequency_data_2 = re.split("\s+",frequency_data)
		frequency_data_3 = STANDARD_TOOLS.NULL_ELIMINATE(frequency_data_2)
		for y in range(len(frequency_array)):
			tmp_frequency_value = frequency_data_3[y]
			frequency_array[y] = tmp_frequency_value
		
		while 1:
			tmp_max_amino_index = int(np.argmax(frequency_array))
			tmp_max_amino_frequency = float(frequency_array[tmp_max_amino_index])
			
			if len(consensus_residue_indexes) == int(0):
				tmp_consensus_amino_index = str(tmp_max_amino_index)
				consensus_residue_indexes.append(tmp_consensus_amino_index)
				frequency_array[tmp_max_amino_index] = float(0)
			elif tmp_max_amino_frequency-frequency_border_line > float(0):
				tmp_consensus_amino_index = str(tmp_max_amino_index)
				consensus_residue_indexes.append(tmp_consensus_amino_index)
				frequency_array[tmp_max_amino_index] = float(0)
			else:
				break
		
		consensus_residue_number = consensus_residue_indexes[np.random.randint(0,len(consensus_residue_indexes))]
		index_to_amino_letter = {"0":"A","1":"C","2":"D","3":"E","4":"F","5":"G","6":"H","7":"I","8":"K","9":"L","10":"M","11":"N","12":"P","13":"Q","14":"R","15":"S","16":"T","17":"V","18":"W","19":"Y"}		
		mutation_residue_number = str(parent_residue_number[mutation_residue_number])
		residue_after_mutation = str(index_to_amino_letter[str(consensus_residue_number)])
		tmp_mutation_information = f"{mutation_residue_number}:{residue_after_mutation}"
		
		return tmp_mutation_information

	def RANDOM(self,parent_sequence,parent_residue_number,mutation_rate,intmsa_data,frequency_border_line):
	# Determine the number of mutations and the residue number to be mutated (Random Mutation).
		
		random_mutation_information = []
#		max_mutation_number = float(len(parent_sequence))*float(mutation_rate)
		max_mutation_number = int(3)
		min_mutation_number = int(0)
		mutation_number = np.random.randint(min_mutation_number,max_mutation_number+1)
		mutation_residue_number = np.zeros(mutation_number,int)
		
		for w in range(len(mutation_residue_number)):
			tmp_mutation_site = np.random.randint(0,len(parent_residue_number))
			mutation_residue_number[w] = tmp_mutation_site
			
		for x in range(len(mutation_residue_number)):
			tmp_random_mutation_residue_number = mutation_residue_number[x]

			tmp_random_mutation_information = self.SELECT_RANDOM_SITE(tmp_random_mutation_residue_number,parent_residue_number,intmsa_data,frequency_border_line)			

			random_mutation_information.append(tmp_random_mutation_information)
		
		return random_mutation_information
		
	def RECOMBINATION(self,directory_name,chain_name):
	# Recombination Mutation.

		parent_candidate_list = os.listdir(f"{directory_name}/temporary_selected")
		parent_candidate_list_2 = STANDARD_TOOLS.NULL_ELIMINATE(parent_candidate_list)
		repeat = int(0)
		max_repeat = int(5)
		recombination_mutation_information = []

		while repeat < max_repeat:
			
			use_parent_candidate = []
			while len(use_parent_candidate) < int(2):
				tmp_candidate_index = np.random.randint(0,len(parent_candidate_list_2))
				if not tmp_candidate_index in use_parent_candidate:
					use_parent_candidate.append(tmp_candidate_index)
			
			template_sequence,template_residue_number,template_pose = STANDARD_TOOLS.PDB_TO_SEQ(f"{directory_name}/temporary_selected/{parent_candidate_list_2[use_parent_candidate[0]]}",chain_name)
			reference_sequence,X_reference_residue_number_X,X_reference_pose_X = STANDARD_TOOLS.PDB_TO_SEQ(f"{directory_name}/temporary_selected/{parent_candidate_list_2[use_parent_candidate[1]]}",chain_name)
			template_pdb = f"{directory_name}/temporary_selected/{parent_candidate_list_2[use_parent_candidate[0]]}"
			mutations_parent_file_index = use_parent_candidate[0]

			mutation_range_max_min = []
			while len(mutation_range_max_min) < int(2):
				tmp_random_range_value = np.random.randint(0,len(template_residue_number))
				if not tmp_random_range_value in mutation_range_max_min:
					mutation_range_max_min.append(tmp_random_range_value)
			mutation_range_max_min_array = np.zeros(len(mutation_range_max_min),int)
			for z in range(len(mutation_range_max_min)):
				mutation_range_max_min_array[z] = int(mutation_range_max_min[z])
			max_mutation_range = np.max(mutation_range_max_min_array)
			min_mutation_range = np.min(mutation_range_max_min_array)
			
			for aa in range(min_mutation_range,max_mutation_range+1):
				tmp_residue_number = int(aa)
				tmp_template_residue = template_sequence[tmp_residue_number]
				tmp_reference_residue = reference_sequence[tmp_residue_number]				
				if tmp_template_residue == tmp_reference_residue:
					continue
				else:
					tmp_recombination_mutation_residue_number = template_residue_number[tmp_residue_number]
					tmp_recombination_mutation_information = f"{tmp_recombination_mutation_residue_number}:{tmp_reference_residue}"
					recombination_mutation_information.append(tmp_recombination_mutation_information)
			
			if len(recombination_mutation_information) > 0:
				break
			else:
				repeat += 1
			
		return recombination_mutation_information,template_pose,template_pdb,mutations_parent_file_index

	def DISULFIDE_CHECK(self,parent_pose,mutations):
	# Delete mutations with "DISULFIDE".

		deletion_list = []
		for bb in range(len(mutations)):
			tmp_mutation_residue_number = int(mutations[bb].split(":")[0])
			mutation_entry_number = str(bb)
			tmp_types = str(parent_pose.residue_type(tmp_mutation_residue_number).variant_types())
			print(tmp_types)
			if re.search("DISULFIDE",tmp_types):
				deletion_list.append(mutation_entry_number)
			
		if len(deletion_list) == int(0):
			return mutations
		elif len(deletion_list) == int(1):
			del mutations[int(deletion_list[0])]
			return mutations
		else:
			deletion_list.reverse()
			for cc in range(len(deletion_list)):
				del mutations[int(deletion_list[cc])]
			return mutations

	def MUTATE_POSE(self,parent_pose,mutations,task_pack_mut):
	# Mutagenesis.

		mutation_pose = Pose()
		mutation_pose.assign(parent_pose)
		
		checked_mutations = self.DISULFIDE_CHECK(parent_pose,mutations)
		
		task_pack_mut = np.zeros(len(checked_mutations),int)
		for dd in range(len(checked_mutations)):
			split_mutation_data = checked_mutations[dd].split(":")
			tmp_mutation_residue_number_data = int(split_mutation_data[0])
			tmp_after_mutation_residue = str(split_mutation_data[1])
			task_pack_mut[dd] = tmp_mutation_residue_number_data
		
			mutate_residue(mutation_pose,tmp_mutation_residue_number_data,tmp_after_mutation_residue)
			
		return mutation_pose,task_pack_mut

	def TOURNAMENT(self,output_child_pdbs,evaluation_scores,directory_name,ff,tournament_entry_number):
	# Select one of the better samples among several randomly selected samples.

		entry_number_list = []
		while len(entry_number_list) < tournament_entry_number:
			tmp_entry_number = np.random.randint(0,len(output_child_pdbs))
			if not tmp_entry_number in entry_number_list:
				entry_number_list.append(tmp_entry_number)
				
		entry_scores_array = np.zeros(tournament_entry_number,float)
		entry_pdbs_list = []
		for gg in range(len(entry_number_list)):
			tmp_score = evaluation_scores[int(entry_number_list[gg])]
			tmp_pdb = output_child_pdbs[int(entry_number_list[gg])]
			entry_scores_array[gg] = tmp_score
			entry_pdbs_list.append(tmp_pdb)

		min_score = np.min(entry_scores_array)
		min_score_index = np.argmin(entry_scores_array)
		winner_pdb = entry_pdbs_list[min_score_index]
		sample_mutations_list = os.listdir(f"{directory_name}/temporary_sample_mutations")
		winner_mutations_parent_file = sample_mutations_list[entry_number_list[min_score_index]]
		os.system(f"cp {directory_name}/temporary_sample_mutations/{winner_mutations_parent_file} {directory_name}/temporary_mutations/{ff+1}.fasta")
		print(min_score)
		print(winner_pdb)
		
		return winner_pdb,min_score

# ===============(Before input.)===============

input_pdb = "NO-INPUT"
input_library_directory = "NO-INPUT"
selected_chain = "NO-INPUT"
number_of_mutations = "UNSPECIFIED"
output_log = "output.log"
generation_number = int(0)
set_repeat_number = int(0)
reu_flag = int(0)# <FITNESS_FUNCTION>-(START)
hisol_flag = int(0)# <FITNESS_FUNCTION>-(GOAL)
total_pressure_count = int(0)
unchange_cut_flag = int(0)
skip_GAOptimizer_flag = int(0)
skip_1_network_flag = int(0)

# ===============(Other input.)===============

rank_flag = "centrality"
centrality_2_flag = "between"
select_flag = "power"
extract_flag = "centrality"
extract_centrality_flag = "between"
step_flag = "below"

# ===============(Input.)===============

while len(sys.argv) > 1:
	if sys.argv[1] == "-PDB":
		input_pdb = sys.argv[2]
		del sys.argv[1:3]
	elif sys.argv[1] == "-DIR":
	# Without "/" at the end.
		input_library_directory = sys.argv[2]
		del sys.argv[1:3]
	elif sys.argv[1] == "-CHAIN":
		selected_chain = str(sys.argv[2])
		del sys.argv[1:3]
	elif sys.argv[1] == "-OUTPUT":
	# No problem without (="output.log").
		output_log = str(sys.argv[2])
		del sys.argv[1:3]
	elif sys.argv[1] == "-GENNUM":
		generation_number = int(sys.argv[2])
		del sys.argv[1:3]
	elif sys.argv[1] == "-RPTNUM":
		set_repeat_number = int(sys.argv[2])
		del sys.argv[1:3]
	elif sys.argv[1] == "-REU":# <FITNESS_FUNCTION>-(START)
		reu_flag = int(1)
		total_pressure_count += int(1)
		del sys.argv[1]
	elif sys.argv[1] == "-HISOL":
		hisol_flag = int(1)# <FITNESS_FUNCTION>-(GOAL)
		total_pressure_count += int(1)
		del sys.argv[1]
	elif sys.argv[1] == "-MUTNUM":
	# Without this tag, the number of mutations is automatically defined by the converged number.
		number_of_mutations = str(sys.argv[2])
		del sys.argv[1:3]
# --------------------------------------
	elif sys.argv[1] == "-UNCHANGECUT":
		unchange_cut_flag = int(1)
		del sys.argv[1]
	elif sys.argv[1] == "-SKIPGAOptimizer":
		skip_GAOptimizer_flag = int(1)
		del sys.argv[1]
	elif sys.argv[1] == "-SKIPNET":
		skip_1_network_flag = int(1)
		del sys.argv[1]

# XXXXXXXXXXXXXXX(Not Recommended)XXXXXXXXXXXXXXX-(START)
	elif sys.argv[1] == "-RANK":
	# "count" / "centrality".
		rank_flag = str(sys.argv[2])
		if rank_flag == "centrality":
		# "between" / "close" / "degree".
			centrality_2_flag = str(sys.argv[3])
			del sys.argv[1:4]
		else:
			del sys.argv[1:3]
	elif sys.argv[1] == "-SELECT":
	# "top" / "power" / "edgecut" / "nodecut".
		select_flag = str(sys.argv[2])
		if select_flag == "power":
		# "degree" / "count" / "centrality".
			extract_flag = str(sys.argv[3])
			if extract_flag == "centrality":
			# "between" / "close" / "degree".
				extract_centrality_flag = str(sys.argv[4])
				del sys.argv[4]
			# "above" / "below".
			step_flag = str(sys.argv[4])
			del sys.argv[1:5]
		else:
			del sys.argv[1:3]
# XXXXXXXXXXXXXXX(Not Recommended)XXXXXXXXXXXXXXX-(GOAL)

# ===============(Error handling of forgotten inputs.)===============

if input_pdb == "NO-INPUT":
	print("INPUT ERROR : Please input 'PDB Name' !")
	sys.exit(1)
elif input_library_directory == "NO-INPUT":
	print("INPUT ERROR : Please input 'Library Path' !")
	sys.exit(1)
elif selected_chain == "NO-INPUT":
	print("INPUT ERROR : Please input 'Chain Name' !")
	sys.exit(1)
elif generation_number == int(0):
	print("INPUT ERROR : Please input 'Generation Number' !")
	sys.exit(1)
elif set_repeat_number == int(0):
	print("INPUT ERROR : Please input 'Set Repeat Number' !")
	sys.exit(1)
elif total_pressure_count == int(0):
	print("INPUT ERROR : Please input at least 1 fitness function !")
	sys.exit(1)

# ===============(Output file tag.)===============

selective_pressure_tag = ""
if reu_flag == int(1):# <FITNESS_FUNCTION>-(START)
	selective_pressure_tag += "RE"
if hisol_flag == int(1):
	selective_pressure_tag += "Hi"# <FITNESS_FUNCTION>-(GOAL)

if rank_flag == "centrality":
	centrality_tag = f"_{centrality_2_flag}"
else:
	centrality_tag = ""

if select_flag == "power":
	step_tag = f"_{step_flag}"
	extract_tag = f"_{extract_flag}"
	if extract_flag == "centrality":
		extract_centrality_tag = f"_{extract_centrality_flag}"
	else:
		extract_centrality_tag = ""
else:
	step_tag = ""
	extract_tag = ""
	extract_centrality_tag = ""

if unchange_cut_flag == int(1):
	unchange_cut_tag = "_unchangecut"
else:
	unchange_cut_tag = ""

tag_set = f"{selective_pressure_tag}_{set_repeat_number}_{generation_number}_{rank_flag}{centrality_tag}_{select_flag}{step_tag}{extract_tag}{extract_centrality_tag}{unchange_cut_tag}"

# ===============(Preparation before GAOptimizer.)===============

stp_residue_from_pdb,stp_residue_number_from_pdb,stp_pose_from_pdb = STANDARD_TOOLS.PDB_TO_SEQ(input_pdb,selected_chain)
stp_pose = stp_pose_from_pdb
sequence_length = len(stp_residue_number_from_pdb)

# >>>>>>>>>>>>>>>VARIABLE<<<<<<<<<<<<<<<

sub_mutations_step = 5# How many steps are changed in the direction of increasing and decreasing the number of mutations, respectively (Mutations' candidates = Max "this number"*2+1) ?
max_number_of_mutations = float(sequence_length/3)# Maximum number of mutations if the number of mutations is not defined in the input.

# ==========(Criteria for the number of mutations.)==========

mutations_number_border_list = []
mutations_number_tag_list = []

if number_of_mutations == "UNSPECIFIED":
	tmp_mutations_number_border = float(sub_mutations_step)
	while tmp_mutations_number_border <= max_number_of_mutations:
		mutations_number_border_list.append(tmp_mutations_number_border)
		mutations_number_tag_list.append(str(int(tmp_mutations_number_border)))
		tmp_mutations_number_border += float(sub_mutations_step)
else:
	if number_of_mutations[-3:] == "per":
		mutations_number_border = sequence_length*float(float(number_of_mutations[:-3])/100)
	else:
		mutations_number_border = float(number_of_mutations)
	mutations_number_border_list.append(mutations_number_border)
	mutations_number_tag_list.append(str(number_of_mutations))

# ==========(Make a list of fitness functions per cycle.)==========

selective_pressure_variety = []
if reu_flag == int(1):# <FITNESS_FUNCTION>-(START)
	selective_pressure_variety.append("REU")
if hisol_flag == int(1):
	selective_pressure_variety.append("HISOL")# <FITNESS_FUNCTION>-(GOAL)
selective_pressure_list = []
for l in range(set_repeat_number):
	for m in range(len(selective_pressure_variety)):
		selective_pressure_list.append(selective_pressure_variety[m])

# ===============(Role of traditional "GAOptimizer".)===============

change_GAOptimizer_flag = int(0)
if (len(selective_pressure_variety) == int(1)) and (set_repeat_number == int(1)):
	change_GAOptimizer_flag = int(1)

# ===============(GAOptimizer main program.)===============

if skip_GAOptimizer_flag == int(0):# If skip, the next step is to create a mutation network.

	# >>>>>>>>>>>>>>>VARIABLE<<<<<<<<<<<<<<<	

	random_mutation_percent = int(30)
	recombination_mutation_percent_increasing = int(2)# Percentage increase per generational evolution.
	sample_number = int(100)# The number of child-generation samples.
	random_mutation_incidence = float(0.015)# A ratio that is multiplied by the length of the sequence to calculate the maximum number of mutations introduced at one time.
	parent_candidate_number = int(sample_number*0.3)# The number of parent-generation samples.
	consensus_frequency_border_line = float(30)# Amino acid frequency recognized as consensus residues (percentage) .
	tournament_entry_number = int(5)
	percent_determine_generation = int(50)# Generation number for which the incidence of random and recombinant mutations is no longer calculated.
	
	# ==========(Preparation of "output.log".)==========
	
	python_file_name = str(sys.argv[0]).split("/")[-1]
	library_file_name = ""
	input_library_file_data = os.listdir(input_library_directory)
	for hh in range(len(input_library_file_data)):
		library_file_name += f"{input_library_file_data[hh]},"
	selective_pressure_number = int(0)
	selective_pressure_data = ""
	if reu_flag == int(1):# <FITNESS_FUNCTION>-(START)
		selective_pressure_number += 1
		selective_pressure_data += "REU or "
	if hisol_flag == int(1):
		selective_pressure_number += 1
		selective_pressure_data += "HISOL or "# <FITNESS_FUNCTION>-(GOAL)
	number_of_generation_number_letter = len(str(generation_number-1))
	
	# ==========(Write design parameters in "output.log".)==========
	
	final_output_data = ""
	final_output_data += "# (parameters)\n"
	final_output_data += f"# Python program                              : {python_file_name}\n"
	final_output_data += f"# Input PDB                                   : {input_pdb}\n"
	final_output_data += f"# Library File                                : {library_file_name[0:-1]}\n"
	final_output_data += f"# Selected Chain                              : {selected_chain}\n"
	final_output_data += f"# Number of Generation                        : {generation_number}\n"
	final_output_data += f"# Number of Cycle Repeat                      : {set_repeat_number}\n"
	final_output_data += f"# Number of Selective Pressure                : {selective_pressure_number}   ( {selective_pressure_data[0:-4]} )\n"
	final_output_data += f"# Number of Sample                            : {sample_number}\n"
	final_output_data += f"# Final Random Mutation Rate (percent)        : {random_mutation_percent}\n"
	final_output_data += f"# Number of Next Generation Parent Candidates : {parent_candidate_number}\n\n"
	final_output_data += f"# [Left] Original Scores of Average in Parent Candidates ( {selective_pressure_data[0:-4]} )\n"
	final_output_data += "# [Center] Energy Values of Elite\n"
	final_output_data += f"# [Right] Original Scores of Elite ( {selective_pressure_data[0:-4]} )\n\n"

	# ==========(Preparation before starting GAOptimizer cycle.)==========

	MAFFTtoINTMSAlign(input_library_directory,stp_residue_from_pdb)
	with open("tmp_intmsa.out","r") as fh_4:
		intmsa_data = fh_4.read().split("\n")
	intmsa_data_2 = STANDARD_TOOLS.NULL_ELIMINATE(intmsa_data)
	del intmsa_data_2[0]
	library_hisol_score = float(0)
	X_mutations_list_X = []
	
	# ==========(GAOptimizer cycle)==========

	for n in range(len(selective_pressure_list)):# Cycle ("Set number" * "the number of selective pressure") .

		tmp_selective_pressure = selective_pressure_list[n]
		
		# =====(If there are directories for each cycle from the previous run, create new directories.)=====

		if n == int(0):
			set_number = int(n)
			directory_list = os.listdir("./")
			for u in directory_list:
				if re.search(r'REU$',u):# <FITNESS_FUNCTION>-(START)
					shutil.rmtree(u)
				elif re.search(r'HISOL$',u):
					shutil.rmtree(u)# <FITNESS_FUNCTION>-(GOAL)
		directory_name = f"{n+1}_{selective_pressure_list[n]}"
		os.mkdir(directory_name)
		os.mkdir(f"{directory_name}/temporary_pyrosetta")
		os.mkdir(f"{directory_name}/temporary_elite")
		os.mkdir(f"{directory_name}/temporary_selected")
		os.mkdir(f"{directory_name}/temporary_mutations")
		os.mkdir(f"{directory_name}/temporary_sample_mutations")
		os.mkdir(f"{directory_name}/temporary_previous_mutations")

		final_output_data += f"\n# {directory_name}\n\n"
	
		for o in range(generation_number):# Generation.
		
			tmp_generation_number = int(o)				
			task_pack_mut = []

			if tmp_generation_number == int(0):
				initial_flag = int(1)
				
				initial_calculation_score = CALCULATION_SCORE(stp_pose_from_pdb,task_pack_mut,tmp_generation_number,tmp_selective_pressure,initial_flag,intmsa_data_2,stp_pose,library_hisol_score)# <FITNESS_FUNCTION>
				# Energy minimize.
				
				initial_flag = int(0)
			
				mutation = MUTATION(random_mutation_percent,tmp_generation_number,recombination_mutation_percent_increasing,sample_number,random_mutation_incidence,parent_candidate_number,directory_name,selected_chain,stp_residue_from_pdb,stp_residue_number_from_pdb,intmsa_data_2,stp_pose_from_pdb,task_pack_mut,tmp_selective_pressure,initial_flag,stp_pose,library_hisol_score,input_pdb,X_mutations_list_X,consensus_frequency_border_line,tournament_entry_number,percent_determine_generation)# <FITNESS_FUNCTION>
				# Mutagenesis.
				
				library_hisol_score = mutation.library_hisol_score

			else:
			
				# =====(Obtain details of the previous generation's elite (overwrite later) .)=====
				
				elite_pdb_list = os.listdir(f"{directory_name}/temporary_elite")
				for ll in range(len(elite_pdb_list)):
					if str(elite_pdb_list[ll]).split('-')[0] == f"{tmp_generation_number-1}":
						tmp_input_pdb = f"{directory_name}/temporary_elite/{elite_pdb_list[ll]}"
				tmp_input_sequence,tmp_input_residue_number,tmp_input_pose = STANDARD_TOOLS.PDB_TO_SEQ(tmp_input_pdb,selected_chain)

				# =====(Initialize used directories.)=====

				if os.path.isdir(f"{directory_name}/temporary_selected"):
					os.system(f"rm -f {directory_name}/temporary_pyrosetta/*")
					os.system(f"rm -f {directory_name}/temporary_previous_mutations/*")
					os.system(f"rm -f {directory_name}/temporary_sample_mutations/*")
				else:
					os.mkdir(f"{directory_name}/temporary_selected")
					os.system(f"rm -f {directory_name}/temporary_pyrosetta/*")
					os.system(f"rm -f {directory_name}/temporary_previous_mutations/*")
					os.system(f"rm -f {directory_name}/temporary_sample_mutations/*")

				initial_flag = int(1)
				
				calculation_score_EM = CALCULATION_SCORE(tmp_input_pose,task_pack_mut,tmp_generation_number,tmp_selective_pressure,initial_flag,intmsa_data_2,stp_pose,library_hisol_score)# <FITNESS_FUNCTION>
				# Energy minimize.

				initial_flag = int(0)
			
				mutation = MUTATION(random_mutation_percent,tmp_generation_number,recombination_mutation_percent_increasing,sample_number,random_mutation_incidence,parent_candidate_number,directory_name,selected_chain,tmp_input_sequence,tmp_input_residue_number,intmsa_data_2,tmp_input_pose,task_pack_mut,tmp_selective_pressure,initial_flag,stp_pose,library_hisol_score,tmp_input_pdb,X_mutations_list_X,consensus_frequency_border_line,tournament_entry_number,percent_determine_generation)# <FITNESS_FUNCTION>
				# Mutagenesis.

			# =====(Write each score for the generation in "output.log".)=====

			tmp_output_selected_average_score = mutation.return_selected_average_score
			tmp_output_elite_energy_value = mutation.return_elite_energy_score
			tmp_output_elite_score = mutation.return_elite_min_score

			final_output_data += "{0} : {1:5.3f}, {2:5.3f}, {3:5.3f}\n".format(str(o).rjust(number_of_generation_number_letter),float(str(tmp_output_selected_average_score).rjust(9)),float(str(tmp_output_elite_energy_value).rjust(9)),float(str(tmp_output_elite_score).rjust(9)))

		# =====(Initialize used directories.)=====

		os.system(f"rm -f {directory_name}/temporary_previous_mutations/*")
		os.system(f"rm -f {directory_name}/temporary_sample_mutations/*")

	# ==========(Final output in Phase.I.)==========

	with open(output_log,"w") as fh_5:
		fh_5.write(final_output_data)

	print("\n   FINISH 'GAOptimizer' !!!\n")

# ********************Phase.II:[Determine mutational combination]********************

class EXTRACT_MUTATION_DATA():
# Extract data on mutations and order of introduction.

	def __init__(self,selective_pressure_list):
		
		self.mutations_list = self.MUTATIONS_DATA(selective_pressure_list)

		self.ORDERS_DATA(self.mutations_list)

	def MUTATIONS_DATA(self,selective_pressure_list):
	# Convert the mutation groups of each generation of each cycle into a triple nested list.

		tmp_final_fh = []
		mutations_list = []
		for ddd in range(len(selective_pressure_list)):# Cycle.

			directory_name_2 = f"{ddd+1}_{selective_pressure_list[ddd]}"
			tmp_final_fh.append(f"fh_final_{ddd+1}")
			with open(f"{directory_name_2}/temporary_mutations/0.fasta","r") as tmp_final_fh[-1]:
				final_mutations_data = tmp_final_fh[-1].read()
			final_mutations_data_2 = final_mutations_data.split("\n")[:-1]

			mutation_set_list = []
			for fff in range(len(final_mutations_data_2)):# Generation.
				mutations_set = final_mutations_data_2[fff].split(":")[1]
				if len(mutations_set) > int(2):
					mutation_list = []

					if re.search(",",mutations_set):
						mutation_list_2 = mutations_set[1:-1].split(", ")
						for nnn in range(len(mutation_list_2)):
							mutations_data = mutation_list_2[nnn].replace("'","")
							append_flag_2 = int(0)
							for ooo in range(len(mutation_set_list)):
								if mutations_data in mutation_set_list[ooo]:# Care for cases where unchanged mutations are selected many times in the same cycle.
									append_flag_2 = int(1)
							if append_flag_2 == int(0):
								mutation_list.append(mutations_data)
						if len(mutation_list) > int(0):
							mutation_set_list.append(mutation_list)
					else:
						mutations_data = mutations_set[1:-1].replace("'","")
						append_flag_3 = int(0)
						for ppp in range(len(mutation_set_list)):
							if mutations_data in mutation_set_list[ppp]:
								append_flag_3 = int(1)
						if append_flag_3 == int(0):
							mutation_list.append(mutations_data)
							mutation_set_list.append(mutation_list)
			mutations_list.append(mutation_set_list)
		
		return mutations_list

	def ORDERS_DATA(self,mutations_list):
	# Add the order number to the list named by the mutation name (from the length of the list, you can derive the number of cycles in which the mutation was adopted).

		mutations_order_data_list = []
		mutations_order_data_name_list = []
		for hhh in range(len(mutations_list)):# Cycle.
			if len(mutations_list[hhh]) > int(0):
				for iii in range(len(mutations_list[hhh])):# Generation.
					for jjj in range(len(mutations_list[hhh][iii])):

						if len(mutations_order_data_list) == int(0):
							mutations_order_data_list.append(mutations_list[hhh][iii][jjj])
							mutations_order_data_name_list.append(mutations_list[hhh][iii][jjj])
							mutations_order_data_list[-1] = []
							mutations_order_data_list[-1].append(str(iii))
						else:
							append_flag = int(0)
							for kkk in range(len(mutations_order_data_name_list)):
								if str(mutations_list[hhh][iii][jjj]) == str(mutations_order_data_name_list[kkk]):
									mutations_order_data_list[kkk].append(str(iii))# Add to existing list.
									append_flag = int(1)
							if append_flag == int(0):
								mutations_order_data_list.append(mutations_list[hhh][iii][jjj])
								mutations_order_data_name_list.append(mutations_list[hhh][iii][jjj])
								mutations_order_data_list[-1] = []
								mutations_order_data_list[-1].append(str(iii))

		output_order = ""
		for lll in range(len(mutations_order_data_list)):
			output_order += f"{mutations_order_data_name_list[lll]}:["
			for mmm in range(len(mutations_order_data_list[lll])):
				output_order += f"{mutations_order_data_list[lll][mmm]},"
			output_order = f"{output_order[:-1]}]\n"
		
		with open("mutations_order.fasta","w") as fh_order:
			fh_order.write(output_order)

		# __________(Output the file as a simple note indicating the order of mutations adopted in each cycle.)__________

		mutations_cycle_data = ""
		for mno in range(len(mutations_list)):
			for nop in range(len(mutations_list[mno])):
				for opq in range(len(mutations_list[mno][nop])):
					mutations_cycle_data += f"{mutations_list[mno][nop][opq]},"
				mutations_cycle_data = f"{mutations_cycle_data[:-1]}\n"
			mutations_cycle_data += "\n"

		with open("tmp_mutations_list.fasta","w") as fh_mutations:
			fh_mutations.write(mutations_cycle_data)

class INITIAL_NETWORK():
# Create a Python file that draws a network where nodes represent mutations and edges represent adoption in the same cycle.

	def __init__(self,mutations_list,selective_pressure_list,reu_flag,hisol_flag,selective_pressure_variety):# <FITNESS_FUNCTION>
	
		# __________(Preparation to run the file and draw the network.)__________

		self.network_file_data = ""
		self.network_file_data += "import numpy as np\nimport os\nimport re\nimport time\nimport networkx as nx\nimport matplotlib.pyplot as plt\n\n"
		self.network_file_data += "start_time_2 = time.time()\n\n"
		self.network_file_data += "plt.figure(figsize = (15,12),layout='tight')\n"
		self.network_file_data += "G = nx.Graph()\n"

		self.reu_nodes_list,self.hisol_nodes_list,self.all_nodes_list,self.edges_data = self.EDGE(mutations_list,selective_pressure_list)# <FITNESS_FUNCTION>

		# __________(Output the mutations adopted in each fitness function.)__________

		if reu_flag == int(1):# <FITNESS_FUNCTION>-(START)
			print(f"REU MUTATIONS         : {self.reu_nodes_list}")
		if hisol_flag == int(1):
			print(f"HiSol Score MUTATIONS : {self.hisol_nodes_list}")# <FITNESS_FUNCTION>-(GOAL)

		self.nodes_data,self.node_colors_data,self.node_sizes_data = self.NODE(selective_pressure_variety,self.reu_nodes_list,self.hisol_nodes_list,self.all_nodes_list,reu_flag,hisol_flag)# <FITNESS_FUNCTION>

		#__________(Write the converted data to the Python file.)__________

		self.edges_data = f"{self.edges_data[:-1]}]"
		self.nodes_data = f"{self.nodes_data[:-1]}]"
		self.edges_count_1 = self.edges_data.count("','")
		print(f"EDGES COUNT : {self.edges_count_1}")
		self.nodes_count_1 = self.nodes_data.count(",")+1
		print(f"NODES COUNT : {self.nodes_count_1}")
		self.node_colors_data = f"{self.node_colors_data[:-1]}]"
		self.node_sizes_data = f"{self.node_sizes_data[:-1]}]"
		
		self.network_file_data += f"{self.nodes_data}\n"
		self.network_file_data += f"{self.node_colors_data}\n"
		self.network_file_data += f"{self.node_sizes_data}\n"
		self.network_file_data += "G.add_nodes_from(nodes)\n"
		self.network_file_data += f"{self.edges_data}\n"
		self.network_file_data += "G.add_edges_from(edges)\n"

		# __________(Preparation to draw the network.)__________
		
		self.network_file_data += "pos = nx.spring_layout(G,k=0.3,scale=18,center=[10,10])\n"
		self.network_file_data += "nx.draw_networkx_edges(G,pos)\n"
		self.network_file_data += "nx.draw_networkx_nodes(G,pos,node_color=node_colors,node_size=node_sizes)\n"
		self.network_file_data += "nx.draw_networkx_labels(G,pos,font_weight='bold')\n"
		self.network_file_data += "plt.axis('off')\nplt.plot(G)\n\n"
		self.network_file_data += "end_time_2 = time.time()\n"
		self.network_file_data += "program_time_2 = end_time_2-start_time_2\n"
		self.network_file_data += "with open('program_time.fasta','a') as fh_time_2:\n"
		self.network_file_data += r"	fh_time_2.write(f'{program_time_2}\n')"

		# >>>>>>>>>>>>>>>VARIABLE<<<<<<<<<<<<<<<

		self.network_file_data += "\n\n#plt.show()\n"# Display the network without "#".

		# __________(Create Python file.)__________

		with open("network_output.py","w") as fh_6:
			fh_6.write(self.network_file_data)

	def EDGE(self,mutations_list,selective_pressure_list):
	# Draw edges between all nodes adopted in the same cycle and classify the nodes for each selection pressure.
		
		reu_nodes_list = [];hisol_nodes_list = [];all_nodes_list = []# <FITNESS_FUNCTION>
		all_edges_list = []
		all_edges = []

		edges_data = "edges = ["

		for pp in range(len(mutations_list)):# Cycle.
			tmp_nodes_list = []
			
			# __________(Classify mutations according to the fitness function adopted.)__________
			
			for qq in range(len(mutations_list[pp])):# Generation.
				for rr in range(len(mutations_list[pp][qq])):

					if selective_pressure_list[pp] == "REU":# <FITNESS_FUNCTION>-(START)
						if not mutations_list[pp][qq][rr] in reu_nodes_list:
							reu_nodes_list.append(f"{mutations_list[pp][qq][rr]}")
					if selective_pressure_list[pp] == "HISOL":
						if not mutations_list[pp][qq][rr] in hisol_nodes_list:
							hisol_nodes_list.append(f"{mutations_list[pp][qq][rr]}")# <FITNESS_FUNCTION>-(GOAL)

					tmp_nodes_list.append(str(mutations_list[pp][qq][rr]))
					all_nodes_list.append(str(mutations_list[pp][qq][rr]))

			# __________(Draw edges between all nodes adopted in the same cycle.)__________

			if len(tmp_nodes_list) > int(1):
				for uu in range(len(tmp_nodes_list)-1):
					for vv in range(uu+1,len(tmp_nodes_list)):
						if str(tmp_nodes_list[uu][:-1]) != str(tmp_nodes_list[vv][:-1]):
							if (not f"('{str(tmp_nodes_list[uu])}','{str(tmp_nodes_list[vv])}')," in all_edges_list) or (not f"('{str(tmp_nodes_list[vv])}','{str(tmp_nodes_list[uu])}')," in all_edges_list):
								edges_data += f"('{str(tmp_nodes_list[uu])}','{str(tmp_nodes_list[vv])}'),"
								all_edges_list.append(f"('{str(tmp_nodes_list[uu])}','{str(tmp_nodes_list[vv])}'),")

							all_edges.append(f"('{str(tmp_nodes_list[uu])}','{str(tmp_nodes_list[vv])}'),")

		# __________(Calculate the number of times edges are drawn as the number of times the mutation pair is adopted.)__________

		all_edges_data = ""
		for tmp_edge_data in range(len(all_edges)):
			tmp_edge_5 = all_edges[tmp_edge_data]
			if not re.search(tmp_edge_5[:-1],all_edges_data):			
				all_edges_data += f"{tmp_edge_5[:-1]}:{all_edges.count(tmp_edge_5)}\n"

		with open("edges_all_count.fasta","w") as fh_edge_count:
			fh_edge_count.write(all_edges_data)

		return reu_nodes_list,hisol_nodes_list,all_nodes_list,edges_data# <FITNESS_FUNCTION>

	def NODE(self,selective_pressure_variety,reu_nodes_list,hisol_nodes_list,all_nodes_list,reu_flag,hisol_flag):# <FITNESS_FUNCTION>
	# Set the color of the node corresponding to the adopted selection pressure and the size of the node corresponding to the adopted number of cycles, respectively.

		nodes_data = "nodes = ["
		node_colors_data = "node_colors = ["
		node_sizes_data = "node_sizes = ["

		if len(selective_pressure_variety) > int(1):
			delete_hisol_nodes_number_list = []# <FITNESS_FUNCTION>-(START)

			for zz in range(len(reu_nodes_list)):
				overlap_flag = int(0)

				for aaa in range(len(hisol_nodes_list)):
					if reu_nodes_list[zz] == hisol_nodes_list[aaa]:
						nodes_data += f"'{str(reu_nodes_list[zz])}',"
						node_colors_data += "'purple',"
						tmp_node_count = all_nodes_list.count(str(reu_nodes_list[zz]))
						node_sizes_data += f"{int(tmp_node_count*200)},"
						delete_hisol_nodes_number_list.append(int(aaa))
						overlap_flag = int(1)

				if overlap_flag == int(0):
					nodes_data += f"'{str(reu_nodes_list[zz])}',"
					node_colors_data += "'red',"
					tmp_node_count = all_nodes_list.count(str(reu_nodes_list[zz]))
					node_sizes_data += f"{int(tmp_node_count*200)},"

			if len(delete_hisol_nodes_number_list) > int(0):
				delete_hisol_nodes_number_list.sort()
				delete_hisol_nodes_number_list.reverse()
				for ccc in range(len(delete_hisol_nodes_number_list)):
					del hisol_nodes_list[delete_hisol_nodes_number_list[ccc]]

			for bbb in range(len(hisol_nodes_list)):
				nodes_data += f"'{str(hisol_nodes_list[bbb])}',"
				node_colors_data += "'blue',"
				tmp_node_count = all_nodes_list.count(str(hisol_nodes_list[bbb]))
				node_sizes_data += f"{int(tmp_node_count*200)},"

		else:
			if reu_flag == int(1):
				for qqq in range(len(reu_nodes_list)):
					nodes_data += f"'{str(reu_nodes_list[qqq])}',"
					node_colors_data += "'red',"
					tmp_node_count = all_nodes_list.count(str(reu_nodes_list[qqq]))
					node_sizes_data += f"{int(tmp_node_count*200)},"
			elif hisol_flag == int(1):
				for rrr in range(len(hisol_nodes_list)):
					nodes_data += f"'{str(hisol_nodes_list[rrr])}',"
					node_colors_data += "'blue',"
					tmp_node_count = all_nodes_list.count(str(hisol_nodes_list[rrr]))
					node_sizes_data += f"{int(tmp_node_count*200)},"# <FITNESS_FUNCTION>-(GOAL)

		return nodes_data,node_colors_data,node_sizes_data

class NETWORK_CUT():
# Reduce the dimension of the network by deleting some of the nodes.

	def __init__(self,stp_residue_from_pdb,unchange_cut_flag):
	
		# __________(Preparation to run the file and draw the network.)__________

		self.make_pdb_data = ""
		self.make_pdb_data += "import numpy as np\nimport os\nimport re\nimport copy\nimport math\nimport time\nimport networkx as nx\nimport matplotlib.pyplot as plt\nimport sys\n\n"
		self.make_pdb_data += "start_time_4 = time.time()\n\n"
		self.make_pdb_data += "plt.figure(figsize = (15,12), layout = 'tight')\n"

		# __________(Read the factors used in the previous network.)__________

		with open("network_output.py","r") as fh_net_2:
			self.network_output_data = fh_net_2.read()

		self.network_output_list = self.network_output_data.split("\n")
		for klm in range(len(self.network_output_list)):
			self.tmp_network_output_data = self.network_output_list[klm].split(" ")[0]
			if self.tmp_network_output_data == "nodes":
				self.nodes_text_data = self.network_output_list[klm]
			elif self.tmp_network_output_data == "node_colors":
				self.node_colors_text_data = self.network_output_list[klm]
			elif self.tmp_network_output_data == "node_sizes":
				self.node_sizes_text_data = self.network_output_list[klm]
			elif self.tmp_network_output_data == "edges":
				self.edges_text_data = self.network_output_list[klm]
				break
				
		self.make_pdb_data += f"{self.nodes_text_data}\n"
		self.make_pdb_data += f"{self.node_colors_text_data}\n"
		self.make_pdb_data += f"{self.node_sizes_text_data}\n"
		self.make_pdb_data += f"{self.edges_text_data}\n"

		# __________(Delete nodes that are mutations that do not change from the original residues.)__________

		if unchange_cut_flag == int(1):

			self.make_pdb_data = self.UNCHANGE_CUT(self.make_pdb_data,stp_residue_from_pdb)

			self.make_pdb_data += "print(f'NODES COUNT AFTER UNCHANGE CUT : {len(nodes)} / NODES LIST AFTER UNCHANGE CUT : {nodes}')\n"
			
			self.make_pdb_data += "if len(nodes) == int(0):\n"
			self.make_pdb_data += r"	print('\n\nNETWORK ERROR : Network was removed by cut ! Therefore, no further operations are possible !\n\n')"
			self.make_pdb_data += "\n	sys.exit(1)\n"				

			self.make_pdb_data += "print(f'EDGES COUNT AFTER UNCHANGE CUT : {len(edges)}')\n\n" 

		# __________(Preparation to draw the network.)__________

		self.make_pdb_data += "G_4 = nx.Graph()\n"
		self.make_pdb_data += "G_4.add_nodes_from(nodes)\n"
		self.make_pdb_data += "G_4.add_edges_from(edges)\n"
		self.make_pdb_data += "pos = nx.spring_layout(G_4,k=0.3,scale=10,center=[10,10])\n"
		self.make_pdb_data += "nx.draw_networkx_nodes(G_4,pos,node_color=node_colors,node_size=node_sizes)\n"
		self.make_pdb_data += "nx.draw_networkx_edges(G_4,pos)\n"
		self.make_pdb_data += "nx.draw_networkx_labels(G_4,pos,font_weight='bold')\n"
		self.make_pdb_data += "plt.axis('off')\n"
		self.make_pdb_data += "plt.plot(G_4)\n\n"
		self.make_pdb_data += "end_time_4 = time.time()\n"
		self.make_pdb_data += "program_time_4 = end_time_4-start_time_4\n"
		self.make_pdb_data += "with open('program_time.fasta','a') as fh_time_4:\n"
		self.make_pdb_data += r"	fh_time_4.write(f'{program_time_4}\n')"

		# >>>>>>>>>>>>>>>VARIABLE<<<<<<<<<<<<<<<

		self.make_pdb_data += "\n\n#plt.show()\n\n"
		
	def UNCHANGE_CUT(self,make_pdb_data,stp_residue_from_pdb):
	# Delete mutations that do not change from the original residues and delete related nodes and edges.

		# __________(Pick mutations of the same residues as the sequence template.)__________

		make_pdb_data += "# __________(Unchange_Cut)__________\n\n"
		make_pdb_data += f"stp_residue_from_pdb = {stp_residue_from_pdb}\n"
		make_pdb_data += "delete_node_indexes_list_2 = []\n"
		make_pdb_data += "delete_edge_indexes_list_2 = []\n\n"
		make_pdb_data += "for tmp_node_2 in range(len(nodes)):\n"
		make_pdb_data += "	tmp_node_residue_number = int(nodes[tmp_node_2][:-1])\n"
		make_pdb_data += "	tmp_node_after_residue = nodes[tmp_node_2][-1]\n"		
		make_pdb_data += "	if stp_residue_from_pdb[tmp_node_residue_number-1] == tmp_node_after_residue:\n"
		make_pdb_data += "		delete_node_indexes_list_2.append(tmp_node_2)\n"

		# __________(Delete edges with these nodes.)__________
		
		make_pdb_data += "delete_node_indexes_list_2.reverse()\n"
		make_pdb_data += "for delete_3 in range(len(delete_node_indexes_list_2)):\n"
		make_pdb_data += "	tmp_delete_node = nodes[delete_node_indexes_list_2[delete_3]]\n"
		make_pdb_data += "	for tmp_edge_2 in range(len(edges)):\n"
		make_pdb_data += "		if re.search(tmp_delete_node,str(edges[tmp_edge_2])):\n"
		make_pdb_data += "			delete_edge_indexes_list_2.append(tmp_edge_2)\n"
		make_pdb_data += "	print(f'DELETE {nodes[delete_node_indexes_list_2[delete_3]]}')\n"
		make_pdb_data += "	del nodes[delete_node_indexes_list_2[delete_3]]\n"
		make_pdb_data += "	del node_colors[delete_node_indexes_list_2[delete_3]]\n"
		make_pdb_data += "	del node_sizes[delete_node_indexes_list_2[delete_3]]\n"
		
		make_pdb_data += "delete_edge_indexes_2 = list(set(delete_edge_indexes_list_2))\n"
		make_pdb_data += "delete_edge_indexes_2.sort()\n"
		make_pdb_data += "delete_edge_indexes_2.reverse()\n"
		make_pdb_data += "for delete_4 in range(len(delete_edge_indexes_2)):\n"
		make_pdb_data += "	del edges[delete_edge_indexes_2[delete_4]]\n"

		# __________(Delete isolated nodes by deleting edges.)__________
		
		make_pdb_data += "delete_node_indexes_list_3 = []\n"
		make_pdb_data += "G_3 = nx.Graph()\n"
		make_pdb_data += "G_3.add_nodes_from(nodes)\n"
		make_pdb_data += "G_3.add_edges_from(edges)\n"
		make_pdb_data += "for tmp_node_4 in range(len(nodes)):\n"
		make_pdb_data += "	if G_3.degree(nodes[tmp_node_4]) == int(0):\n"
		make_pdb_data += "		delete_node_indexes_list_3.append(tmp_node_4)\n"
		make_pdb_data += "delete_node_indexes_list_3.reverse()\n"
		make_pdb_data += "for delete_5 in range(len(delete_node_indexes_list_3)):\n"
		make_pdb_data += "	del nodes[delete_node_indexes_list_3[delete_5]]\n"
		make_pdb_data += "	del node_colors[delete_node_indexes_list_3[delete_5]]\n"
		make_pdb_data += "	del node_sizes[delete_node_indexes_list_3[delete_5]]\n\n"
		
		return make_pdb_data

class SELECT_MUTATIONS():
# Determine multiple mutation group candidates.

	def __init__(self,make_pdb_data,rank_flag,select_flag,centrality_2_flag,extract_flag,extract_centrality_flag,step_flag,mutations_number_border,sub_mutations_step,tag_set,mutations_number_tag):

		self.make_pdb_data = make_pdb_data
		self.make_pdb_data += f"start_time_5_{int(mutations_number_border)} = time.time()\n\n"

		# __________(Create a file with some of the mutations saved in ranking order.)__________

		now_directory_lineup = os.listdir("./")
		sub_fasta_flag = int(0)
		for sub_fasta in range(len(now_directory_lineup)):
			if now_directory_lineup[sub_fasta] == f"sub_mutations_list_sel{mutations_number_tag}.fasta":
				sub_fasta_flag = int(1)
		if sub_fasta_flag == int(1):
			os.system(f"rm sub_mutations_list_sel{mutations_number_tag}.fasta")
		fh_sub_mut_list = []
		fh_sub_mut_list.append(f"fh_sub_mut_{int(mutations_number_border)}")
		with open(f"sub_mutations_list_sel{mutations_number_tag}.fasta","w") as fh_sub_mut_list[-1]:
			fh_sub_mut_list[-1].write("")

		self.make_pdb_data += f"sub_mutations_step = int({sub_mutations_step})\n"

		# __________(Reduce the network to a scale-free network (Section.A) .)__________

		self.make_pdb_data += "nodes_for_select = copy.deepcopy(nodes)\n"

		if select_flag == "power":# Make scale-free network (power index) .

			self.make_pdb_data = self.POWER_INDEX(self.make_pdb_data,extract_flag,extract_centrality_flag,step_flag,mutations_number_border)

			# >>>>>>>>>>>>>>>VARIABLE<<<<<<<<<<<<<<<

			self.make_pdb_data += "\n#plt.show()\n\n"# Display the finished scale-free network.

			# _____(Handover to the next process.)_____

			self.make_pdb_data += "nodes_for_select = copy.deepcopy(new_nodes_list)\n"			
			self.graph_name = "graph_list[-1]"

		# XXXXXXXXXXXXXXX(Not Recommended)XXXXXXXXXXXXXXX-(START)
		else:
			self.graph_name = "G_4"
		# XXXXXXXXXXXXXXX(Not Recommended)XXXXXXXXXXXXXXX-(GOAL)

		# __________(Section.E: Mutational ranking criteria.)__________

		# XXXXXXXXXXXXXXX(Not Recommended)XXXXXXXXXXXXXXX-(START)
		if rank_flag == "count":

			self.make_pdb_data = self.COUNT_RANK(self.make_pdb_data)
		
		# XXXXXXXXXXXXXXX(Not Recommended)XXXXXXXXXXXXXXX-(GOAL)

		elif rank_flag == "centrality":

			self.make_pdb_data = self.CENTRALITY_RANK(centrality_2_flag,self.make_pdb_data,self.graph_name)
		
		else:
			print('INPUT ERROR : Input "count" or "centrality" for -RANK !')
			sys.exit(1)

		# __________(Section.A: Mutational selective algorithm.)__________

		if (select_flag == "top") or (select_flag == "power"):

			self.make_pdb_data = self.TOP_NUMBER(self.make_pdb_data,mutations_number_border,mutations_number_tag)

		# XXXXXXXXXXXXXXX(Not Recommended)XXXXXXXXXXXXXXX-(START)
		elif select_flag == "edgecut":

			self.make_pdb_data = self.EDGECUT_COMMUNITY(self.make_pdb_data,mutations_number_border,mutations_number_tag)

		elif select_flag == "nodecut":

			self.make_pdb_data = self.NODECUT_COMMUNITY(self.make_pdb_data,centrality_2_flag,rank_flag,mutations_number_border,mutations_number_tag)
		
		else:
			print('INPUT ERROR : Input "top" or "power" or "edgecut" or "nodecut" for -SELECT !')
			sys.exit(1)

		# XXXXXXXXXXXXXXX(Not Recommended)XXXXXXXXXXXXXXX-(GOAL)

		# __________(Sort the current mutations by residue number.)__________

		self.make_pdb_data += "selected_number = []\n"
		self.make_pdb_data += "selected_mutations_2 = []\n"
		self.make_pdb_data += "for pqr in range(len(selected_mutations)):\n"
		self.make_pdb_data += "	tmp_selected_number = int(selected_mutations[pqr][:-1])\n"
		self.make_pdb_data += "	selected_number.append(tmp_selected_number)\n"
		self.make_pdb_data += "while len(selected_mutations_2) != len(selected_number):\n"
		self.make_pdb_data += "	tmp_min_selected_number_index = np.argmin(selected_number)\n"
		self.make_pdb_data += "	selected_mutations_2.append(selected_mutations[tmp_min_selected_number_index])\n"
		self.make_pdb_data += "	selected_number[tmp_min_selected_number_index] = int(100000)\n"				
		self.make_pdb_data += "print(f'ORIGINAL MUTATIONS : {selected_mutations_2}')\n"

		# __________(Output a file written the selected mutations.)__________

		self.make_pdb_data += "selected_mutations_data = ', '.join(selected_mutations_2)\n"
		self.make_pdb_data += "selected_mutations_data_2 = f'[{selected_mutations_data}]'\n"

		self.make_pdb_data += f"with open('select_mutations_{tag_set}_sel{mutations_number_tag}.fasta','w') as fh_select_1_{int(mutations_number_border)}:\n"
		self.make_pdb_data += f"	fh_select_1_{int(mutations_number_border)}.write(selected_mutations_data_2)\n\n"

		self.make_pdb_data += f"end_time_5_{int(mutations_number_border)} = time.time()\n"
		self.make_pdb_data += f"program_time_5 = end_time_5_{int(mutations_number_border)}-start_time_5_{int(mutations_number_border)}\n"
		self.make_pdb_data += "fh_time_5_list = []\n"
		self.make_pdb_data += f"fh_time_5_list.append('fh_time_5_{int(mutations_number_border)}')\n"
		self.make_pdb_data += "with open('program_time.fasta','a') as fh_time_5_list[-1]:\n"
		self.make_pdb_data += r"	fh_time_5_list[-1].write(f'{program_time_5}\n')"
		self.make_pdb_data += "\n"

	# XXXXXXXXXXXXXXX(Not Recommended)XXXXXXXXXXXXXXX-(START)
	def COUNT_RANK(self,make_pdb_data):
	# The higher the number of cycles in which the mutation is adopted, the higher the ranking (Section.E) .

		make_pdb_data += "with open('mutations_order.fasta','r') as fh_order_nodes_for_select:\n"
		make_pdb_data += "	mutations_order_data_1 = fh_order_nodes_for_select.read()\n"
		make_pdb_data += r"mutations_order_list_1 = mutations_order_data_1.split('\n')[:-1]"

		make_pdb_data += "\nmutations_count_list = []\n"
		make_pdb_data += "for tmp_node_3 in range(len(nodes_for_select)):\n"
		make_pdb_data += "	for order_data in range(len(mutations_order_list_1)):\n"
		make_pdb_data += """		if nodes_for_select[tmp_node_3] == f'{mutations_order_list_1[order_data].split(":")[0]}':\n"""
		make_pdb_data += "			tmp_order_list = mutations_order_list_1[order_data].split(':')[1][1:-1].split(',')\n"
		make_pdb_data += "			tmp_count_data = len(tmp_order_list)\n"
		make_pdb_data += "			mutations_count_list.append(tmp_count_data)\n"

		make_pdb_data += "values_for_rank = copy.deepcopy(mutations_count_list)\n"
		make_pdb_data += """print(f'RANK for "count" : {values_for_rank}')\n\n"""

		return make_pdb_data
	# XXXXXXXXXXXXXXX(Not Recommended)XXXXXXXXXXXXXXX-(GOAL)

	def CENTRALITY_RANK(self,centrality_2_flag,make_pdb_data,graph_name):
	# The more central the network, the higher its ranking (Section.E) .

		# __________(Section.F: Type of centrality.)__________
		
		# XXXXXXXXXXXXXXX(Not Recommended)XXXXXXXXXXXXXXX-(START)
		if centrality_2_flag == "degree":# The more edges a node has, the higher the centrality of that node (Section.F) .

			make_pdb_data += f"centrality_values = list(nx.degree_centrality({graph_name}).values())\n"
		# XXXXXXXXXXXXXXX(Not Recommended)XXXXXXXXXXXXXXX-(GOAL)
		
		elif centrality_2_flag == "between":# The more times a node is passed in the shortest path between all pairs of nodes, the higher the centrality of that node (Section.F) .

			make_pdb_data += f"centrality_values = list(nx.betweenness_centrality({graph_name}).values())\n"

		# XXXXXXXXXXXXXXX(Not Recommended)XXXXXXXXXXXXXXX-(START)
		elif centrality_2_flag == "close":# The smaller the average of the shortest paths from one node to another, the higher the centrality of that node (Section.F) .

			make_pdb_data += f"centrality_values = list(nx.closeness_centrality({graph_name}).values())\n"
		# XXXXXXXXXXXXXXX(Not Recommended)XXXXXXXXXXXXXXX-(GOAL)
		
		else:
			print('INPUT ERROR : Input "between" or "close" or "degree" for centrality !')
			sys.exit(1)

		make_pdb_data += "values_for_rank = copy.deepcopy(centrality_values)\n"
		make_pdb_data += """print(f'RANK for "centrality" : {values_for_rank}')\n\n"""
		
		return make_pdb_data

	def TOP_NUMBER(self,make_pdb_data,mutations_number_border,mutations_number_tag):
	# Select a defined number of mutations from the top of the ranking (Section.A) .

		make_pdb_data += "selected_mutations = []\n"
		make_pdb_data += "end_flag_for_top = int(0)\n"

		# __________(Select nodes to a defined number of mutations.)__________

		make_pdb_data += f"if len(nodes_for_select) > {mutations_number_border}:\n"
		make_pdb_data += "	while end_flag_for_top == int(0):\n"

		# _____(Extract temporary top tie nodes.)_____

		make_pdb_data += "		tmp_append_selected_mutations = []\n"
		make_pdb_data += "		tmp_max_index = np.argmax(values_for_rank)\n"
		make_pdb_data += "		tmp_max_value = values_for_rank[tmp_max_index]\n"
		make_pdb_data += "		for tmp_node_value in range(len(values_for_rank)):\n"
		make_pdb_data += "			if tmp_max_value == values_for_rank[tmp_node_value]:\n"
		make_pdb_data += "				tmp_append_selected_mutations.append(nodes_for_select[tmp_node_value])\n"
		make_pdb_data += "				values_for_rank[tmp_node_value] = -100000\n"

		make_pdb_data += "		sub_mutations_data = ''\n"

		# _____(Add mutations because the number of mutations does not reach the defined number.)_____

		make_pdb_data += f"		if int(len(selected_mutations))+int(len(tmp_append_selected_mutations)) <= {mutations_number_border}:\n"
		make_pdb_data += "			for tmp_append_value_node in range(len(tmp_append_selected_mutations)):\n"
		make_pdb_data += "				selected_mutations.append(tmp_append_selected_mutations[tmp_append_value_node])\n"
		make_pdb_data += "				sub_mutations_data += f'{tmp_append_selected_mutations[tmp_append_value_node]},'\n"

		make_pdb_data += "			fh_sub_mut_add_list = []\n"
		make_pdb_data += f"			fh_sub_mut_add_list.append('fh_sub_mut_add_{int(mutations_number_border)}')\n"
		make_pdb_data += f"			with open('sub_mutations_list_sel{mutations_number_tag}.fasta','a') as fh_sub_mut_add_list[-1]:\n"
		make_pdb_data += r"				fh_sub_mut_add_list[-1].write(f'{sub_mutations_data[:-1]}\n')"

		# _____(When a defined number can be reached, evaluate which is closer to that number before or after the addition.)_____

		make_pdb_data += "\n		else:\n"

		# ___(Before the addition.)___

		make_pdb_data += f"			if abs((int(len(selected_mutations))+int(len(tmp_append_selected_mutations)))-{mutations_number_border}) < abs(int(len(selected_mutations))-{mutations_number_border}):\n"

		make_pdb_data += "				for tmp_append_value_node_2 in range(len(tmp_append_selected_mutations)):\n"
		make_pdb_data += "					selected_mutations.append(tmp_append_selected_mutations[tmp_append_value_node_2])\n"
		make_pdb_data += "					sub_mutations_data += f'{tmp_append_selected_mutations[tmp_append_value_node_2]},'\n"

		make_pdb_data += "				fh_sub_mut_add_list = []\n"
		make_pdb_data += f"				fh_sub_mut_add_list.append('fh_sub_mut_add_{int(mutations_number_border)}')\n"
		make_pdb_data += f"				with open('sub_mutations_list_sel{mutations_number_tag}.fasta','a') as fh_sub_mut_add_list[-1]:\n"
		make_pdb_data += r"					fh_sub_mut_add_list[-1].write(f'{sub_mutations_data[:-1]}\n')"
		make_pdb_data += "\n"
		make_pdb_data += r"					fh_sub_mut_add_list[-1].write('beforeafter\n')"

		# ___(After the addition.)___

		make_pdb_data += "\n			else:\n"
		make_pdb_data += "				for tmp_append_value_node_3 in range(len(tmp_append_selected_mutations)):\n"
		make_pdb_data += "					sub_mutations_data += f'{tmp_append_selected_mutations[tmp_append_value_node_3]},'\n"

		make_pdb_data += "				fh_sub_mut_add_list = []\n"
		make_pdb_data += f"				fh_sub_mut_add_list.append('fh_sub_mut_add_{int(mutations_number_border)}')\n"
		make_pdb_data += f"				with open('sub_mutations_list_sel{mutations_number_tag}.fasta','a') as fh_sub_mut_add_list[-1]:\n"
		make_pdb_data += r"					fh_sub_mut_add_list[-1].write('beforeafter\n')"
		make_pdb_data += "\n"
		make_pdb_data += r"					fh_sub_mut_add_list[-1].write(f'{sub_mutations_data[:-1]}\n')"
		make_pdb_data += "\n				sub_mutations_step += -1\n"

		# ___(After determining the mutation groups that are closer to the defined number, information about the adoption candidate is collected.)___

		make_pdb_data += "			for sub_step in range(sub_mutations_step):\n"
		make_pdb_data += "				sub_mutations_data = ''\n"
		make_pdb_data += "				tmp_max_index = np.argmax(values_for_rank)\n"
		make_pdb_data += "				tmp_max_value = values_for_rank[tmp_max_index]\n"
		make_pdb_data += "				if tmp_max_value == -100000:\n"
		make_pdb_data += "					break\n"
		make_pdb_data += "				for tmp_node_value in range(len(values_for_rank)):\n"
		make_pdb_data += "					if tmp_max_value == values_for_rank[tmp_node_value]:\n"
		make_pdb_data += "						sub_mutations_data += f'{nodes_for_select[tmp_node_value]},'\n"
		make_pdb_data += "						values_for_rank[tmp_node_value] = -100000\n"

		make_pdb_data += "				fh_sub_mut_add_list = []\n"
		make_pdb_data += f"				fh_sub_mut_add_list.append('fh_sub_mut_add_{int(mutations_number_border)}')\n"
		make_pdb_data += f"				with open('sub_mutations_list_sel{mutations_number_tag}.fasta','a') as fh_sub_mut_add_list[-1]:\n"
		make_pdb_data += r"					fh_sub_mut_add_list[-1].write(f'{sub_mutations_data[:-1]}\n')"

		make_pdb_data += "\n			end_flag_for_top = int(1)\n"# End "while" cycle.
		
		# __________(If the number of all nodes is less than the defined number of mutations.)__________

		make_pdb_data += "else:\n"
		make_pdb_data += "	selected_mutations = copy.deepcopy(nodes_for_select)\n"

		make_pdb_data += "	fh_sub_mut_add_list = []\n"
		make_pdb_data += f"	fh_sub_mut_add_list.append('fh_sub_mut_add_{int(mutations_number_border)}')\n"
		make_pdb_data += f"	with open('sub_mutations_list_sel{mutations_number_tag}.fasta','a') as fh_sub_mut_add_list[-1]:\n"
		make_pdb_data += r"		fh_sub_mut_add_list[-1].write('beforeafter\n')"

		# ___(Information about the adoption order is collected.)___

		make_pdb_data += "\n	for sub_step in range(sub_mutations_step):\n"
		make_pdb_data += "		sub_mutations_data = ''\n"
		make_pdb_data += "		tmp_max_index = np.argmax(values_for_rank)\n"
		make_pdb_data += "		tmp_max_value = values_for_rank[tmp_max_index]\n"
		make_pdb_data += "		if tmp_max_value == -100000:\n"
		make_pdb_data += "			break\n"
		make_pdb_data += "		for tmp_node_value in range(len(values_for_rank)):\n"
		make_pdb_data += "			if tmp_max_value == values_for_rank[tmp_node_value]:\n"
		make_pdb_data += "				sub_mutations_data += f'{nodes_for_select[tmp_node_value]},'\n"
		make_pdb_data += "				values_for_rank[tmp_node_value] = -100000\n"

		make_pdb_data += "		fh_sub_mut_add_list = []\n"
		make_pdb_data += f"		fh_sub_mut_add_list.append('fh_sub_mut_add_{int(mutations_number_border)}')\n"		
		make_pdb_data += f"		with open('sub_mutations_list_sel{mutations_number_tag}.fasta','a') as fh_sub_mut_add_list[-1]:\n"
		make_pdb_data += r"			fh_sub_mut_add_list[-1].write(f'{sub_mutations_data[:-1]}\n')"
		make_pdb_data += "\n"

		return make_pdb_data

	def POWER_INDEX(self,make_pdb_data,extract_flag,extract_centrality_flag,step_flag,mutations_number_border):
	# Reduce the network by removing some of the nodes and edges until a scale-free network is achieved (Section.A) .
	
		# __________(Section.B: Ranking criteria based on the selection of nodes to be removed to build a scale-free network.)__________
	
		# XXXXXXXXXXXXXXX(Not Recommended)XXXXXXXXXXXXXXX-(START)
		if extract_flag == "degree":# The number of edges that the node has (Section.B) .

			make_pdb_data += "degree_count_list = []\n"
			make_pdb_data += "for tmp_node_5 in range(len(nodes)):\n"
			make_pdb_data += "	tmp_degree = G_4.degree(nodes[tmp_node_5])\n"
			make_pdb_data += "	degree_count_list.append(tmp_degree)\n"
			make_pdb_data += "extract_value_list = degree_count_list\n"
		# XXXXXXXXXXXXXXX(Not Recommended)XXXXXXXXXXXXXXX-(GOAL)

		elif extract_flag == "centrality":# (Section.B) .

			# __________(Section.C: Types of centrality in ranking criteria for building scale-free networks.)__________

			# XXXXXXXXXXXXXXX(Not Recommended)XXXXXXXXXXXXXXX-(START)
			if extract_centrality_flag == "degree":# (Section.C)

				make_pdb_data += "extract_centrality_values = list(nx.degree_centrality(G_4).values())\n"
			# XXXXXXXXXXXXXXX(Not Recommended)XXXXXXXXXXXXXXX-(GOAL)

			elif extract_centrality_flag == "between":# (Section.C)

				make_pdb_data += "extract_centrality_values = list(nx.betweenness_centrality(G_4).values())\n"

			# XXXXXXXXXXXXXXX(Not Recommended)XXXXXXXXXXXXXXX-(START)
			elif extract_centrality_flag == "close":# (Section.C)

				make_pdb_data += "extract_centrality_values = list(nx.closeness_centrality(G_4).values())\n"
			# XXXXXXXXXXXXXXX(Not Recommended)XXXXXXXXXXXXXXX-(GOAL)
			
			else:
				print('INPUT ERROR : Input "between" or "close" or "degree" for centrality !')
				sys.exit(1)

			make_pdb_data += "extract_value_list = extract_centrality_values\n"

		# XXXXXXXXXXXXXXX(Not Recommended)XXXXXXXXXXXXXXX-(START)
		elif extract_flag == "count":# (Section.B) .

			make_pdb_data = self.COUNT_RANK(make_pdb_data)

			make_pdb_data += "extract_value_list = values_for_rank\n"
		# XXXXXXXXXXXXXXX(Not Recommended)XXXXXXXXXXXXXXX-(GOAL)
		
		else:
			print('INPUT ERROR : Input "degree" or "count" or "centrality" for power !')
			sys.exit(1)

		make_pdb_data += "node_all_rank = np.zeros(len(nodes),int)\n"		
		make_pdb_data += "extract_rank_number = int(1)\n"

		# __________(Replace numbers with rankings.)__________

		make_pdb_data += "while np.count_nonzero(node_all_rank) != len(node_all_rank):\n"
		make_pdb_data += "	tmp_plus_number = int(0)\n"
		make_pdb_data += "	tmp_max_node_index = np.argmax(extract_value_list)\n"
		make_pdb_data += "	tmp_max_value_count = extract_value_list[tmp_max_node_index]\n"
		make_pdb_data += "	for tmp_values in range(len(extract_value_list)):\n"
		make_pdb_data += "		if extract_value_list[tmp_values] == tmp_max_value_count:\n"
		make_pdb_data += "			node_all_rank[tmp_values] = extract_rank_number\n"
		make_pdb_data += "			extract_value_list[tmp_values] = int(-100000)\n"
		make_pdb_data += "			tmp_plus_number += int(1)\n"

		make_pdb_data += "	extract_rank_number += tmp_plus_number\n"

		# __________(Complete the scale-free network (Section.D) .)__________

		if step_flag == "below":

			make_pdb_data += "node_all_rank_list = []\n"
			make_pdb_data += "for node_rank in range(len(node_all_rank)):\n"
			make_pdb_data += "	node_all_rank_list.append(node_all_rank[node_rank])\n"

		make_pdb_data = self.POWER_INDEX_DETERMINE_NETWORK(make_pdb_data,step_flag,mutations_number_border)

		# __________(Preparation to draw the network.)__________

		make_pdb_data += "plt.figure(figsize = (15,12), layout = 'tight')\n"
		make_pdb_data += "pos = nx.spring_layout(graph_list[-1],k=0.3,scale=10,center=[10,10])\n"
		make_pdb_data += "nx.draw_networkx_nodes(graph_list[-1],pos,node_color=new_node_colors_list,node_size=new_node_sizes_list)\n"
		make_pdb_data += "nx.draw_networkx_edges(graph_list[-1],pos)\n"
		make_pdb_data += "nx.draw_networkx_labels(graph_list[-1],pos,font_weight='bold')\n"
		make_pdb_data += "plt.axis('off')\n"
		make_pdb_data += "plt.plot(graph_list[-1])\n\n"

		make_pdb_data += "print(f'NODES IN THE SCALE-FREE NETWORK : {len(new_nodes_list)}')\n"
		make_pdb_data += "print(f'EDGES IN THE SCALE-FREE NETWORK : {len(new_edges_list)}')\n"

		return make_pdb_data

	def POWER_INDEX_DETERMINE_NETWORK(self,make_pdb_data,step_flag,mutations_number_border):
	# Calculations to evaluate the scale-free nature of the rebuild network.
		
		# __________(Preparation to evaluate the scale-free.)__________

		make_pdb_data += "minus_ganma = float(0)\n"
		make_pdb_data += "number_tag = int(1)\n"
		make_pdb_data += "success_flag = int(0)\n"
		make_pdb_data += "graph_list = []\n"
		make_pdb_data += "degree_rank_array_list = []\n"
		make_pdb_data += "degree_list_list = []\n"
		make_pdb_data += "x_array_list = []\ny_array_list = []\n"
		
		# __________(Section.D: The order in which nodes are selected to rebuild the network.)__________

		# XXXXXXXXXXXXXXX(Not Recommended)XXXXXXXXXXXXXXX-(START)
		if step_flag == "above":
		
			make_pdb_data = self.POWER_INDEX_ABOVE(make_pdb_data,mutations_number_border)

		# XXXXXXXXXXXXXXX(Not Recommended)XXXXXXXXXXXXXXX-(GOAL)

		elif step_flag == "below":

			make_pdb_data = self.POWER_INDEX_BELOW(make_pdb_data)
		
		else:
			print('INPUT ERROR : Input "above" or "below" for power !')
			sys.exit(1)

		# __________(Preparation to count degree.)__________

		make_pdb_data += "		x_value_list = []\n"
		make_pdb_data += "		y_value_list = []\n"
		make_pdb_data += "		graph_list.append(f'G_power_{number_tag}')\n"
		make_pdb_data += "		graph_list[-1] = nx.Graph()\n"
		make_pdb_data += "		graph_list[-1].add_nodes_from(new_nodes_list)\n"
		make_pdb_data += "		graph_list[-1].add_edges_from(new_edges_list)\n"
		make_pdb_data += "		degree_list_list.append(f'degree_list_{number_tag}')\n"
		make_pdb_data += "		degree_list_list[-1] = []\n"

		# __________(Count degree.)__________

		make_pdb_data += "		for tmp_node_8 in range(len(new_nodes_list)):\n"
		make_pdb_data += "			tmp_degree_value = graph_list[-1].degree(new_nodes_list[tmp_node_8])\n"
		make_pdb_data += "			if tmp_degree_value > int(0):\n"
		make_pdb_data += "				degree_list_list[-1].append(tmp_degree_value)\n"
		make_pdb_data += "		tmp_degree_list = copy.deepcopy(degree_list_list[-1])\n"
		make_pdb_data += "		degree_rank_array_list.append(f'degree_rank_array_{number_tag}')\n"
		make_pdb_data += "		degree_rank_array_list[-1] = np.zeros(len(degree_list_list[-1]),int)\n"

		# __________(Replace numbers with rankings.)__________

		make_pdb_data += "		rank_number_2 = int(1)\n"
		make_pdb_data += "		while np.count_nonzero(degree_rank_array_list[-1]) != len(degree_rank_array_list[-1]):\n"
		make_pdb_data += "			tmp_plus_number_2 = int(0)\n"
		make_pdb_data += "			tmp_max_node_index_2 = np.argmax(tmp_degree_list)\n"
		make_pdb_data += "			tmp_max_degree_count_2 = tmp_degree_list[tmp_max_node_index_2]\n"
		make_pdb_data += "			if tmp_max_degree_count_2 > int(0):\n"
		make_pdb_data += "				for tmp_degree_3 in range(len(tmp_degree_list)):\n"
		make_pdb_data += "					if tmp_degree_list[tmp_degree_3] == tmp_max_degree_count_2:\n"
		make_pdb_data += "						degree_rank_array_list[-1][tmp_degree_3] = rank_number_2\n"
		make_pdb_data += "						tmp_degree_list[tmp_degree_3] = int(-100000)\n"
		make_pdb_data += "						tmp_plus_number_2 += int(1)\n"
		make_pdb_data += "				rank_number_2 += tmp_plus_number_2\n"
		make_pdb_data += "			else:\n"
		make_pdb_data += "				break\n"

		# __________(Draw a straight line using logarithms and extract slope.)__________

		make_pdb_data += "		for tmp_degree_value_2 in range(len(degree_list_list[-1])):\n"			
		make_pdb_data += "			tmp_x_value = math.log10(int(degree_list_list[-1][tmp_degree_value_2]))\n"
		make_pdb_data += "			tmp_y_value = math.log10(int(degree_rank_array_list[-1][tmp_degree_value_2]))\n"
		make_pdb_data += "			x_value_list.append(tmp_x_value)\n"
		make_pdb_data += "			y_value_list.append(tmp_y_value)\n"

		make_pdb_data += "		if len(x_value_list) > int(0):\n"
		make_pdb_data += "			if x_value_list.count(x_value_list[0]) != len(x_value_list):\n"
		make_pdb_data += "				x_array_list.append(f'x_value_array_{number_tag}')\n"
		make_pdb_data += "				y_array_list.append(f'y_value_array_{number_tag}')\n"
		make_pdb_data += "				x_array_list[-1] = np.zeros(len(x_value_list),float)\n"
		make_pdb_data += "				y_array_list[-1] = np.zeros(len(y_value_list),float)\n"

		make_pdb_data += "				for array_value in range(len(x_value_list)):\n"
		make_pdb_data += "					x_array_list[-1][array_value] = float(x_value_list[array_value])\n"
		make_pdb_data += "					y_array_list[-1][array_value] = float(y_value_list[array_value])\n"

		make_pdb_data += "				minus_ganma,X_b_X = np.polyfit(x_array_list[-1],y_array_list[-1],1)\n"
		make_pdb_data += "				print(f'TEMPORARY MINUS GANMA : {minus_ganma}')\n"
		
		make_pdb_data += "				if -3 <= minus_ganma <= -2:\n"
		make_pdb_data += "					success_flag = int(1)\n"

		make_pdb_data += "		number_tag += int(1)\n"

		make_pdb_data += "	else:\n"
		make_pdb_data += "		break\n"

		make_pdb_data += "if success_flag == int(1):\n"
		make_pdb_data += "	print(f'FINAL POWER INDEX : {minus_ganma}')\n"
		make_pdb_data += "elif success_flag == int(0):\n"
		make_pdb_data += "	print('Since a scale-free network could not be created, the original network was restored.')\n"

		# __________(If a scale-free network cannot be constructed with "below", the original network is used as the result, not the scale-free network. (Section.D) .)__________

		if step_flag == "below":

			make_pdb_data += "if len(node_all_rank_list) == int(0):\n"
			make_pdb_data += "	new_nodes_list = copy.deepcopy(nodes)\n"
			make_pdb_data += "	new_node_colors_list = copy.deepcopy(node_colors)\n"
			make_pdb_data += "	new_node_sizes_list = copy.deepcopy(node_sizes)\n"
			make_pdb_data += "	new_edges_list = copy.deepcopy(edges)\n"	
			make_pdb_data += "	graph_list.append(f'G_power_below_0')\n"
			make_pdb_data += "	graph_list[-1] = nx.Graph()\n"
			make_pdb_data += "	graph_list[-1].add_nodes_from(new_nodes_list)\n"
			make_pdb_data += "	graph_list[-1].add_edges_from(new_edges_list)\n"
		
		return make_pdb_data

	# XXXXXXXXXXXXXXX(Not Recommended)XXXXXXXXXXXXXXX-(START)
	def POWER_INDEX_ABOVE(self,make_pdb_data,mutations_number_border):
	# Extract the high-ranked nodes and associated edges and rebuild the network with them (Section.D) .

		make_pdb_data += "new_nodes_list = []\nnew_node_colors_list = []\nnew_node_sizes_list = []\nnew_edges_list = []\n"

		# __________(Start cycle.)__________

		make_pdb_data += f"while ((minus_ganma > -2) or (minus_ganma < -3)) or (len(new_nodes_list) < {mutations_number_border}):\n"

		# _____(Extract temporary top tie nodes.)_____

		make_pdb_data += "	tmp_append_node_list = []\n"
		make_pdb_data += "	tmp_append_edge_list = []\n"
		make_pdb_data += "	tmp_rank_index = np.argmin(node_all_rank)\n"
		make_pdb_data += "	tmp_rank = node_all_rank[tmp_rank_index]\n"

		make_pdb_data += "	if tmp_rank <= len(node_all_rank):\n"
		make_pdb_data += "		for tmp_node_6 in range(len(node_all_rank)):\n"
		make_pdb_data += "			if node_all_rank[tmp_node_6] == tmp_rank:\n"
		make_pdb_data += "				tmp_append_node_list.append(tmp_node_6)\n"

		make_pdb_data += "		for tmp_append_node in range(len(tmp_append_node_list)):\n"
		make_pdb_data += "			new_nodes_list.append(nodes[tmp_append_node_list[tmp_append_node]])\n"
		make_pdb_data += "			new_node_colors_list.append(node_colors[tmp_append_node_list[tmp_append_node]])\n"
		make_pdb_data += "			new_node_sizes_list.append(node_sizes[tmp_append_node_list[tmp_append_node]])\n"

		# _____(Extract relevant edges.)_____

		make_pdb_data += "			for tmp_node_7 in range(len(new_nodes_list)-1):\n"
		make_pdb_data += "				for tmp_edge_3 in range(len(edges)):\n"
		make_pdb_data += """					if (f"('{nodes[tmp_append_node_list[tmp_append_node]]}', '{new_nodes_list[tmp_node_7]}')" == f"{edges[tmp_edge_3]}") or (f"('{new_nodes_list[tmp_node_7]}', '{nodes[tmp_append_node_list[tmp_append_node]]}')" == f"{edges[tmp_edge_3]}"):\n"""
		make_pdb_data += "						new_edges_list.append(edges[tmp_edge_3])\n"

		make_pdb_data += "			node_all_rank[tmp_append_node_list[tmp_append_node]] = int(100000)\n"

		return make_pdb_data
	# XXXXXXXXXXXXXXX(Not Recommended)XXXXXXXXXXXXXXX-(GOAL)

	def POWER_INDEX_BELOW(self,make_pdb_data):
	# Remove low-ranked nodes and associated edges from the original network (Section.D) .
	
		make_pdb_data += "new_nodes_list_list = []\nnew_node_colors_list_list = []\nnew_node_sizes_list_list = []\nnew_edges_list_list = []\n"
		make_pdb_data += "graph_list_isolated = []\n"

		# __________(Start cycle.)__________

		make_pdb_data += "while ((minus_ganma > -2) or (minus_ganma < -3)) and (len(node_all_rank_list) > 0):\n"

		# _____(Takeover of previous data.)_____

		make_pdb_data += "	new_nodes_list_list.append(f'new_nodes_list_{number_tag}')\n"
		make_pdb_data += "	new_node_colors_list_list.append(f'new_node_colors_list_{number_tag}')\n"
		make_pdb_data += "	new_node_sizes_list_list.append(f'new_node_sizes_list_{number_tag}')\n"
		make_pdb_data += "	new_edges_list_list.append(f'new_edges_list_{number_tag}')\n"

		make_pdb_data += "	if number_tag == int(1):\n"
		make_pdb_data += "		new_nodes_list_list[-1] = copy.deepcopy(nodes)\n"
		make_pdb_data += "		new_node_colors_list_list[-1] = copy.deepcopy(node_colors)\n"
		make_pdb_data += "		new_node_sizes_list_list[-1] = copy.deepcopy(node_sizes)\n"
		make_pdb_data += "		new_edges_list_list[-1] = copy.deepcopy(edges)\n"
		make_pdb_data += "	else:\n"
		make_pdb_data += "		new_nodes_list_list[-1] = copy.deepcopy(new_nodes_list_list[-2])\n"
		make_pdb_data += "		new_node_colors_list_list[-1] = copy.deepcopy(new_node_colors_list_list[-2])\n"
		make_pdb_data += "		new_node_sizes_list_list[-1] = copy.deepcopy(new_node_sizes_list_list[-2])\n"
		make_pdb_data += "		new_edges_list_list[-1] = copy.deepcopy(new_edges_list_list[-2])\n"

		# _____(Delete low-ranked nodes and associated edges.)_____

		make_pdb_data += "	tmp_delete_node_list = []\n"
		make_pdb_data += "	tmp_delete_edge_list = []\n"
		make_pdb_data += "	tmp_max_rank_index = np.argmax(node_all_rank_list)\n"
		make_pdb_data += "	tmp_max_rank = node_all_rank_list[tmp_max_rank_index]\n"
		make_pdb_data += "	for tmp_rank in range(len(node_all_rank_list)):\n"
		make_pdb_data += "		if node_all_rank_list[tmp_rank] == tmp_max_rank:\n"
		make_pdb_data += "			tmp_delete_node_list.append(tmp_rank)\n"
		make_pdb_data += "	tmp_delete_node_list.reverse()\n"

		make_pdb_data += "	for tmp_delete_node_2 in range(len(tmp_delete_node_list)):\n"
		make_pdb_data += "		tmp_delete_node_3 = new_nodes_list_list[-1][tmp_delete_node_list[tmp_delete_node_2]]\n"
		make_pdb_data += "		for tmp_edge_4 in range(len(new_edges_list_list[-1])):\n"
		make_pdb_data += "			if re.search(f'{tmp_delete_node_3}',str(new_edges_list_list[-1][tmp_edge_4])):\n"
		make_pdb_data += "				tmp_delete_edge_list.append(tmp_edge_4)\n"
		make_pdb_data += "		del new_nodes_list_list[-1][tmp_delete_node_list[tmp_delete_node_2]]\n"
		make_pdb_data += "		del new_node_colors_list_list[-1][tmp_delete_node_list[tmp_delete_node_2]]\n"
		make_pdb_data += "		del new_node_sizes_list_list[-1][tmp_delete_node_list[tmp_delete_node_2]]\n"
		make_pdb_data += "		del node_all_rank_list[tmp_delete_node_list[tmp_delete_node_2]]\n"

		make_pdb_data += "	tmp_delete_edge_list_2 = list(set(tmp_delete_edge_list))\n"
		make_pdb_data += "	tmp_delete_edge_list_2.sort()\n"
		make_pdb_data += "	tmp_delete_edge_list_2.reverse()\n"
		make_pdb_data += "	for tmp_delete_edge in range(len(tmp_delete_edge_list_2)):\n"
		make_pdb_data += "		del new_edges_list_list[-1][tmp_delete_edge_list_2[tmp_delete_edge]]\n"

		# _____(Delete nodes that has no edges.)_____

		make_pdb_data += "	delete_isolated_node_list = []\n"
		make_pdb_data += "	graph_list_isolated.append(f'graph_isolated_{number_tag}')\n"
		make_pdb_data += "	graph_list_isolated[-1] = nx.Graph()\n"
		make_pdb_data += "	graph_list_isolated[-1].add_nodes_from(new_nodes_list_list[-1])\n"
		make_pdb_data += "	graph_list_isolated[-1].add_edges_from(new_edges_list_list[-1])\n"
		make_pdb_data += "	for tmp_node_9 in range(len(new_nodes_list_list[-1])):\n"
		make_pdb_data += "		if graph_list_isolated[-1].degree(new_nodes_list_list[-1][tmp_node_9]) == int(0):\n"
		make_pdb_data += "			delete_isolated_node_list.append(tmp_node_9)\n"
		make_pdb_data += "	delete_isolated_node_list.reverse()\n"

		make_pdb_data += "	for delete_isolated_node in range(len(delete_isolated_node_list)):\n"
		make_pdb_data += "		del new_nodes_list_list[-1][delete_isolated_node_list[delete_isolated_node]]\n"
		make_pdb_data += "		del new_node_colors_list_list[-1][delete_isolated_node_list[delete_isolated_node]]\n"
		make_pdb_data += "		del new_node_sizes_list_list[-1][delete_isolated_node_list[delete_isolated_node]]\n"
		make_pdb_data += "		del node_all_rank_list[delete_isolated_node_list[delete_isolated_node]]\n"

		# _____(Update the lists.)_____

		make_pdb_data += "	new_nodes_list = new_nodes_list_list[-1]\n"
		make_pdb_data += "	new_node_colors_list = new_node_colors_list_list[-1]\n"
		make_pdb_data += "	new_node_sizes_list = new_node_sizes_list_list[-1]\n"
		make_pdb_data += "	new_edges_list = new_edges_list_list[-1]\n"
		make_pdb_data += "	if tmp_max_rank != int(0):\n"
		
		return make_pdb_data

	# XXXXXXXXXXXXXXX(Not Recommended)XXXXXXXXXXXXXXX-(START)
	def EDGECUT_COMMUNITY(self,make_pdb_data,mutations_number_border,mutations_number_tag):
	# By removing low-ranked edges, the number of nodes is reduced until the network reaches the number of input nodes (Section.A) .
	
		# __________(The number of mutation pairs adopted is read from the file as the number of edges.)__________
		
		make_pdb_data += "neo_edges_list = []\n"
		make_pdb_data += "neo_edges_count_list = []\n"

		make_pdb_data += "with open('edges_all_count.fasta','r') as fh_edge_count_read:\n"
		make_pdb_data += "	edges_count_data = fh_edge_count_read.read()\n"
		make_pdb_data += r"edges_count_data_list = edges_count_data.split('\n')[:-1]"

		# _____(Composite the edges of the same meaning.)_____

		make_pdb_data += "\nwhile len(edges_count_data_list) != int(0):\n"

		make_pdb_data += "	tmp_edge_count_data = edges_count_data_list[0]\n"
		make_pdb_data += "	tmp_edge_data_2 = tmp_edge_count_data.split(':')[0]\n"
		make_pdb_data += "	tmp_1_node = tmp_edge_data_2.split(',')[0][2:-1]\n"
		make_pdb_data += "	tmp_2_node = tmp_edge_data_2.split(',')[1][1:-2]\n"
		make_pdb_data += "	tmp_edge_count = int(0)\n"
		make_pdb_data += "	delete_edge_data_list = []\n"
		make_pdb_data += "	for tmp_edge_data_line in range(len(edges_count_data_list)):\n"
		make_pdb_data += """		if re.search((f"('{tmp_1_node}','{tmp_2_node}')" or f"('{tmp_2_node}','{tmp_1_node}')"),edges_count_data_list[tmp_edge_data_line].split(":")[0]):\n"""
		make_pdb_data += "			tmp_edge_count += int(edges_count_data_list[tmp_edge_data_line].split(':')[1])\n"
		make_pdb_data += "			delete_edge_data_list.append(tmp_edge_data_line)\n"

		make_pdb_data += """	neo_edges_list.append(f"('{tmp_1_node}','{tmp_2_node}')")\n"""
		make_pdb_data += "	neo_edges_count_list.append(tmp_edge_count)\n"
		make_pdb_data += "	delete_edge_data_list.reverse()\n"
		make_pdb_data += "	for tmp_edge_data_index in range(len(delete_edge_data_list)):\n"
		make_pdb_data += "		del edges_count_data_list[delete_edge_data_list[tmp_edge_data_index]]\n"

		# __________(Preparation of cycle.)__________

		make_pdb_data += "edges_for_select = copy.deepcopy(edges)\n"
		make_pdb_data += "end_flag_for_edgecut = int(0)\n"
		make_pdb_data += "graph_list_edgecut = []\n"
		make_pdb_data += "number_tag_edgecut = int(0)\n"

		# __________(Start cycle.)__________

		make_pdb_data += "while end_flag_for_edgecut == int(0):\n"

		make_pdb_data += "	number_tag_edgecut += int(1)\n"
		make_pdb_data += "	tmp_cut_edges_list = []\n"

		# _____(Remove the edges with the lowest number of adoptions.)_____

		make_pdb_data += "	tmp_min_count_index = np.argmin(neo_edges_count_list)\n"
		make_pdb_data += "	tmp_min_count = neo_edges_count_list[tmp_min_count_index]\n"
		make_pdb_data += "	for tmp_count_index in range(len(neo_edges_count_list)):\n"
		make_pdb_data += "		if neo_edges_count_list[tmp_count_index] == tmp_min_count:\n"
		make_pdb_data += "			tmp_cut_edges_list.append(tmp_count_index)\n"
		make_pdb_data += "	tmp_cut_edges_list.reverse()\n"
		make_pdb_data += "	tmp_cut_edges_index_list = []\n"

		make_pdb_data += "	for cut_edge in range(len(tmp_cut_edges_list)):\n"
		make_pdb_data += "		tmp_cut_edge_2 = neo_edges_list[tmp_cut_edges_list[cut_edge]]\n"
		make_pdb_data += "		tmp_cut_1_node = tmp_cut_edge_2.split(',')[0][2:-1]\n"
		make_pdb_data += "		tmp_cut_2_node = tmp_cut_edge_2.split(',')[1][1:-2]\n"
		make_pdb_data += "		for tmp_edge_6 in range(len(edges_for_select)):\n"
		make_pdb_data += """			if re.search((f"('{tmp_cut_1_node}', '{tmp_cut_2_node}')" or f"('{tmp_cut_2_node}', '{tmp_cut_1_node}')"),str(edges_for_select[tmp_edge_6])):\n"""
		make_pdb_data += "				tmp_cut_edges_index_list.append(tmp_edge_6)\n"
		make_pdb_data += "		del neo_edges_list[tmp_cut_edges_list[cut_edge]]\n"
		make_pdb_data += "		del neo_edges_count_list[tmp_cut_edges_list[cut_edge]]\n"

		make_pdb_data += "	tmp_cut_edges_index_list.sort()\n"
		make_pdb_data += "	tmp_cut_edges_index_list.reverse()\n"
		make_pdb_data += "	for tmp_cut_edge_3 in range(len(tmp_cut_edges_index_list)):\n"
		make_pdb_data += "		del edges_for_select[tmp_cut_edges_index_list[tmp_cut_edge_3]]\n"

		# _____(Collect nodes that has no edges.)_____

		make_pdb_data += "	previous_nodes_list = copy.deepcopy(nodes_for_select)\n"
		make_pdb_data += "	graph_list_edgecut.append(f'graph_edgecut_{number_tag_edgecut}')\n"
		make_pdb_data += "	graph_list_edgecut[-1] = nx.Graph()\n"
		make_pdb_data += "	graph_list_edgecut[-1].add_nodes_from(nodes_for_select)\n"
		make_pdb_data += "	graph_list_edgecut[-1].add_edges_from(edges_for_select)\n"
		make_pdb_data += "	tmp_cut_nodes_index_list = []\n"
		make_pdb_data += "	for tmp_node_10 in range(len(nodes_for_select)):\n"
		make_pdb_data += "		if graph_list_edgecut[-1].degree(nodes_for_select[tmp_node_10]) == int(0):\n"
		make_pdb_data += "			tmp_cut_nodes_index_list.append(tmp_node_10)\n"

		make_pdb_data += "	sub_mutations_data = ''\n"

		# _____(Whether the number of nodes remaining after deleting these nodes is greater than the defined number.)_____

		make_pdb_data += f"	if int(len(nodes_for_select)-len(tmp_cut_nodes_index_list)) > {mutations_number_border}:\n"

		make_pdb_data += "		tmp_cut_nodes_index_list.reverse()\n"
		make_pdb_data += "		for tmp_cut_node_3 in range(len(tmp_cut_nodes_index_list)):\n"
		make_pdb_data += "			sub_mutations_data += f'{nodes_for_select[tmp_cut_nodes_index_list[tmp_cut_node_3]]},'\n"
		make_pdb_data += "			del nodes_for_select[tmp_cut_nodes_index_list[tmp_cut_node_3]]\n"

		make_pdb_data += "		fh_sub_mut_add_list = []\n"
		make_pdb_data += f"		fh_sub_mut_add_list.append('fh_sub_mut_add_{int(mutations_number_border)}')\n"
		make_pdb_data += f"		with open('sub_mutations_list_sel{mutations_number_tag}.fasta','a') as fh_sub_mut_add_list[-1]:\n"
		make_pdb_data += r"			fh_sub_mut_add_list[-1].write(f'{sub_mutations_data[:-1]}\n')"

		# _____(If the number of remaining nodes is less than or equal to the defined number.)_____

		make_pdb_data += "\n	else:\n"
		make_pdb_data += "		beforeafter_flag = int(0)\n"
		make_pdb_data += "		if int(len(nodes_for_select)-len(tmp_cut_nodes_index_list)) != int(0):\n"

		# ___(Evaluate which is closer to defined number before or after the deletion.)___

		make_pdb_data += f"			if abs(int(len(nodes_for_select)-len(tmp_cut_nodes_index_list))-{mutations_number_border}) <= abs(len(nodes_for_select)-{mutations_number_border}):\n"
		make_pdb_data += "				tmp_cut_nodes_index_list.reverse()\n"
		make_pdb_data += "				for tmp_cut_node_3 in range(len(tmp_cut_nodes_index_list)):\n"
		make_pdb_data += "					sub_mutations_data += f'{nodes_for_select[tmp_cut_nodes_index_list[tmp_cut_node_3]]},'\n"
		make_pdb_data += "					del nodes_for_select[tmp_cut_nodes_index_list[tmp_cut_node_3]]\n"

		make_pdb_data += "				fh_sub_mut_add_list = []\n"
		make_pdb_data += f"				fh_sub_mut_add_list.append('fh_sub_mut_add_{int(mutations_number_border)}')\n"
		make_pdb_data += f"				with open('sub_mutations_list_sel{mutations_number_tag}.fasta','a') as fh_sub_mut_add_list[-1]:\n"
		make_pdb_data += r"					fh_sub_mut_add_list[-1].write(f'{sub_mutations_data[:-1]}\n')"
		make_pdb_data += "\n"
		make_pdb_data += r"					fh_sub_mut_add_list[-1].write('beforeafter\n')"
		make_pdb_data += "\n				beforeafter_flag = int(1)\n"

		# ___(If the nodes have not been deleted, these nodes are considered future candidates.)___

		make_pdb_data += "		selected_mutations = copy.deepcopy(nodes_for_select)\n"
		make_pdb_data += "		nodes_for_select_2 = copy.deepcopy(nodes_for_select)\n"

		make_pdb_data += "		if beforeafter_flag == int(0):\n"
		make_pdb_data += "			tmp_cut_nodes_index_list.reverse()\n"
		make_pdb_data += "			for tmp_cut_node_3 in range(len(tmp_cut_nodes_index_list)):\n"
		make_pdb_data += "				sub_mutations_data += f'{nodes_for_select_2[tmp_cut_nodes_index_list[tmp_cut_node_3]]},'\n"
		make_pdb_data += "				del nodes_for_select_2[tmp_cut_nodes_index_list[tmp_cut_node_3]]\n"		

		make_pdb_data += "			fh_sub_mut_add_list = []\n"
		make_pdb_data += f"			fh_sub_mut_add_list.append('fh_sub_mut_add_{int(mutations_number_border)}')\n"
		make_pdb_data += f"			with open('sub_mutations_list_sel{mutations_number_tag}.fasta','a') as fh_sub_mut_add_list[-1]:\n"
		make_pdb_data += r"				fh_sub_mut_add_list[-1].write('beforeafter\n')"
		make_pdb_data += "\n"
		make_pdb_data += r"				fh_sub_mut_add_list[-1].write(f'{sub_mutations_data[:-1]}\n')"
		make_pdb_data += "\n			sub_mutations_step += -1\n"

		# ___(Extract future candidates.)___

		make_pdb_data += "		for sub_step in range(sub_mutations_step):\n"
		make_pdb_data += "			sub_mutations_data = ''\n"
		make_pdb_data += "			if len(nodes_for_select_2) != int(0):\n"
		make_pdb_data += "				number_tag_edgecut += int(1)\n"

		# _(Remove the edges with the lowest number of adoptions.)_

		make_pdb_data += "				tmp_cut_edges_list = []\n"
		make_pdb_data += "				tmp_min_count_index = np.argmin(neo_edges_count_list)\n"
		make_pdb_data += "				tmp_min_count = neo_edges_count_list[tmp_min_count_index]\n"
		make_pdb_data += "				for tmp_count_index in range(len(neo_edges_count_list)):\n"
		make_pdb_data += "					if neo_edges_count_list[tmp_count_index] == tmp_min_count:\n"
		make_pdb_data += "						tmp_cut_edges_list.append(tmp_count_index)\n"
		make_pdb_data += "				tmp_cut_edges_list.reverse()\n"
		make_pdb_data += "				tmp_cut_edges_index_list = []\n"

		make_pdb_data += "				for cut_edge in range(len(tmp_cut_edges_list)):\n"
		make_pdb_data += "					tmp_cut_edge_2 = neo_edges_list[tmp_cut_edges_list[cut_edge]]\n"
		make_pdb_data += "					tmp_cut_1_node = tmp_cut_edge_2.split(',')[0][2:-1]\n"
		make_pdb_data += "					tmp_cut_2_node = tmp_cut_edge_2.split(',')[1][1:-2]\n"
		make_pdb_data += "					for tmp_edge_6 in range(len(edges_for_select)):\n"
		make_pdb_data += """						if re.search((f"('{tmp_cut_1_node}', '{tmp_cut_2_node}')" or f"('{tmp_cut_2_node}', '{tmp_cut_1_node}')"),str(edges_for_select[tmp_edge_6])):\n"""
		make_pdb_data += "							tmp_cut_edges_index_list.append(tmp_edge_6)\n"
		make_pdb_data += "					del neo_edges_list[tmp_cut_edges_list[cut_edge]]\n"
		make_pdb_data += "					del neo_edges_count_list[tmp_cut_edges_list[cut_edge]]\n"

		make_pdb_data += "				tmp_cut_edges_index_list.sort()\n"
		make_pdb_data += "				tmp_cut_edges_index_list.reverse()\n"
		make_pdb_data += "				for tmp_cut_edge_3 in range(len(tmp_cut_edges_index_list)):\n"
		make_pdb_data += "					del edges_for_select[tmp_cut_edges_index_list[tmp_cut_edge_3]]\n"

		# _(Collect nodes that has no edges.)_

		make_pdb_data += "				previous_nodes_list = copy.deepcopy(nodes_for_select_2)\n"
		make_pdb_data += "				graph_list_edgecut.append(f'graph_edgecut_{number_tag_edgecut}')\n"
		make_pdb_data += "				graph_list_edgecut[-1] = nx.Graph()\n"
		make_pdb_data += "				graph_list_edgecut[-1].add_nodes_from(nodes_for_select_2)\n"
		make_pdb_data += "				graph_list_edgecut[-1].add_edges_from(edges_for_select)\n"
		make_pdb_data += "				tmp_cut_nodes_index_list = []\n"
		make_pdb_data += "				for tmp_node_10 in range(len(nodes_for_select_2)):\n"
		make_pdb_data += "					if graph_list_edgecut[-1].degree(nodes_for_select_2[tmp_node_10]) == int(0):\n"
		make_pdb_data += "						tmp_cut_nodes_index_list.append(tmp_node_10)\n"

		make_pdb_data += "				tmp_cut_nodes_index_list.reverse()\n"
		make_pdb_data += "				for tmp_cut_node_3 in range(len(tmp_cut_nodes_index_list)):\n"
		make_pdb_data += "					sub_mutations_data += f'{nodes_for_select_2[tmp_cut_nodes_index_list[tmp_cut_node_3]]},'\n"
		make_pdb_data += "					del nodes_for_select_2[tmp_cut_nodes_index_list[tmp_cut_node_3]]\n"

		make_pdb_data += "				fh_sub_mut_add_list = []\n"
		make_pdb_data += f"				fh_sub_mut_add_list.append('fh_sub_mut_add_{int(mutations_number_border)}')\n"
		make_pdb_data += f"				with open('sub_mutations_list_sel{mutations_number_tag}.fasta','a') as fh_sub_mut_add_list[-1]:\n"
		make_pdb_data += r"					fh_sub_mut_add_list[-1].write(f'{sub_mutations_data[:-1]}\n')"

		make_pdb_data += "\n			else:\n"
		make_pdb_data += "				break\n"
		make_pdb_data += "		end_flag_for_edgecut = int(1)\n"

		# __________(Create a file of mutation adoption records.)__________

		make_pdb_data += f"with open('sub_mutations_list_sel{mutations_number_tag}.fasta','r') as fh_sub_mut_read_{int(mutations_number_border)}:\n"
		make_pdb_data += f"	sub_mut_reverse_data = fh_sub_mut_read_{int(mutations_number_border)}.read()\n"
		make_pdb_data += r"sub_mut_reverse_data_2 = sub_mut_reverse_data.split('\n')[:-1]"
		make_pdb_data += "\nprint(sub_mut_reverse_data_2)\n"

		make_pdb_data += "\nsub_mut_forward_data = []\n"
		make_pdb_data += "for sub_reverse_data in range(len(sub_mut_reverse_data_2)):\n"
		make_pdb_data += "	sub_mut_forward_data.append(sub_mut_reverse_data_2[int('-'+f'{int(sub_reverse_data)+1}')])\n"
		make_pdb_data += "print(f'MUTATIONAL ADOPTION REPORT : {sub_mut_forward_data}')\n"
		make_pdb_data += r"sub_mut_forward_data_2 = '\n'.join(sub_mut_forward_data)"

		make_pdb_data += f"\nos.system('rm sub_mutations_list_sel{mutations_number_tag}.fasta')\n"
		make_pdb_data += f"with open('sub_mutations_list_sel{mutations_number_tag}.fasta','w') as fh_sub_mut_w_{int(mutations_number_border)}:\n"
		make_pdb_data += f"	fh_sub_mut_w_{int(mutations_number_border)}.write(sub_mut_forward_data_2)\n"

		return make_pdb_data

	def NODECUT_COMMUNITY(self,make_pdb_data,centrality_2_flag,rank_flag,mutations_number_border,mutations_number_tag):
	# By removing low-ranked nodes, the number of nodes is reduced until the network reaches the number of input nodes (Section.A) .

		# __________(Preparation of cycle.)__________

		make_pdb_data += "edges_for_select = copy.deepcopy(edges)\n"
		make_pdb_data += "end_flag_for_nodecut = int(0)\n"
		make_pdb_data += "cut_graph_list = []\n"
		make_pdb_data += "cut_number_tag = int(1)\n"

		# __________(Start cycle.)__________

		make_pdb_data += "while end_flag_for_nodecut == int(0):\n"

		# _____(Collect nodes with low values.)_____

		make_pdb_data += "	tmp_cut_nodes_list = []\n"
		make_pdb_data += "	tmp_cut_edges_list = []\n"
		make_pdb_data += "	tmp_min_index = np.argmin(values_for_rank)\n"
		make_pdb_data += "	tmp_min_value = values_for_rank[tmp_min_index]\n"
		make_pdb_data += "	for tmp_value_2 in range(len(values_for_rank)):\n"
		make_pdb_data += "		if tmp_min_value == values_for_rank[tmp_value_2]:\n"
		make_pdb_data += "			tmp_cut_nodes_list.append(tmp_value_2)\n"

		make_pdb_data += "	sub_mutations_data = ''\n"

		# _____(Whether the number of nodes remaining after deleting these nodes is greater than the defined number.)_____

		make_pdb_data += f"	if len(nodes_for_select)-len(tmp_cut_nodes_list) > {mutations_number_border}:\n"

		# ___(Delete these nodes and associated edges.)___

		make_pdb_data += "		tmp_cut_nodes_list.sort()\n"
		make_pdb_data += "		tmp_cut_nodes_list.reverse()\n"
		make_pdb_data += "		for tmp_cut_node_index in range(len(tmp_cut_nodes_list)):\n"
		make_pdb_data += "			tmp_cut_node = nodes_for_select[tmp_cut_nodes_list[tmp_cut_node_index]]\n"
		make_pdb_data += "			for tmp_cut_edge_index in range(len(edges_for_select)):\n"
		make_pdb_data += "				if re.search(f'{tmp_cut_node}',str(edges_for_select[tmp_cut_edge_index])):\n"
		make_pdb_data += "					tmp_cut_edges_list.append(tmp_cut_edge_index)\n"
		make_pdb_data += "			sub_mutations_data += f'{nodes_for_select[tmp_cut_nodes_list[tmp_cut_node_index]]},'\n"
		make_pdb_data += "			del nodes_for_select[tmp_cut_nodes_list[tmp_cut_node_index]]\n"
		make_pdb_data += "			del values_for_rank[tmp_cut_nodes_list[tmp_cut_node_index]]\n"

		make_pdb_data += "		tmp_cut_edges_list_2 = list(set(tmp_cut_edges_list))\n"
		make_pdb_data += "		tmp_cut_edges_list_2.sort()\n"
		make_pdb_data += "		tmp_cut_edges_list_2.reverse()\n"
		make_pdb_data += "		for tmp_cut_edge in range(len(tmp_cut_edges_list_2)):\n"
		make_pdb_data += "			del edges_for_select[tmp_cut_edges_list_2[tmp_cut_edge]]\n"

		# ___(Delete nodes that has no edges.)___

		make_pdb_data += "		cut_graph_list.append(f'graph_{cut_number_tag}')\n"
		make_pdb_data += "		cut_graph_list[-1] = nx.Graph()\n"
		make_pdb_data += "		cut_graph_list[-1].add_nodes_from(nodes_for_select)\n"
		make_pdb_data += "		cut_graph_list[-1].add_edges_from(edges_for_select)\n"
		make_pdb_data += "		cut_number_tag += 1\n"

		make_pdb_data += "		tmp_cut_nodes_list_2 = []\n"
		make_pdb_data += "		for tmp_isolated_node_index in range(len(nodes_for_select)):\n"
		make_pdb_data += "			if cut_graph_list[-1].degree(nodes_for_select[tmp_isolated_node_index]) == int(0):\n"
		make_pdb_data += "				tmp_cut_nodes_list_2.append(tmp_isolated_node_index)\n"

		make_pdb_data += "		tmp_cut_nodes_list_2.reverse()\n"
		make_pdb_data += "		for tmp_isolated_node in range(len(tmp_cut_nodes_list_2)):\n"
		make_pdb_data += "			sub_mutations_data += f'{nodes_for_select[tmp_cut_nodes_list_2[tmp_isolated_node]]},'\n"
		make_pdb_data += "			del nodes_for_select[tmp_cut_nodes_list_2[tmp_isolated_node]]\n"
		make_pdb_data += "			del values_for_rank[tmp_cut_nodes_list_2[tmp_isolated_node]]\n"

		# ___(Update the network and recalculate if the criterion is centrality.)___

		make_pdb_data += "		cut_graph_list.append(f'graph_{cut_number_tag}')\n"
		make_pdb_data += "		cut_graph_list[-1] = nx.Graph()\n"
		make_pdb_data += "		cut_graph_list[-1].add_nodes_from(nodes_for_select)\n"
		make_pdb_data += "		cut_graph_list[-1].add_edges_from(edges_for_select)\n"
		make_pdb_data += "		cut_number_tag += 1\n"

		if rank_flag == "centrality":

			if centrality_2_flag == "degree":
				make_pdb_data += "		centrality_values = list(nx.degree_centrality(cut_graph_list[-1]).values())\n"
			elif centrality_2_flag == "between":
				make_pdb_data += "		centrality_values = list(nx.betweenness_centrality(cut_graph_list[-1]).values())\n"
			elif centrality_2_flag == "close":
				make_pdb_data += "		centrality_values = list(nx.closeness_centrality(cut_graph_list[-1]).values())\n"

			make_pdb_data += "		values_for_rank = centrality_values\n"

		make_pdb_data += "		print(f'TEMPORARY VALUES : {values_for_rank}')\n"

		make_pdb_data += "		fh_sub_mut_add_list = []\n"
		make_pdb_data += f"		fh_sub_mut_add_list.append('fh_sub_mut_add_{int(mutations_number_border)}')\n"
		make_pdb_data += f"		with open('sub_mutations_list_sel{mutations_number_tag}.fasta','a') as fh_sub_mut_add_list[-1]:\n"
		make_pdb_data += r"			fh_sub_mut_add_list[-1].write(f'{sub_mutations_data[:-1]}\n')"

		# _____(If the number of remaining nodes is less than or equal to the defined number.)_____

		make_pdb_data += "\n	else:\n"
		make_pdb_data += "		beforeafter_flag = int(0)\n"
		make_pdb_data += "		if int(len(nodes_for_select)-len(tmp_cut_nodes_list)) != int(0):\n"

		# ___(Evaluate which is closer to defined number before or after the deletion.)___

		make_pdb_data += f"			if abs(int(len(nodes_for_select)-len(tmp_cut_nodes_list))-{mutations_number_border}) <= abs(len(nodes_for_select)-{mutations_number_border}):\n"

		# _(Delete these nodes and associated edges, then also isolated nodes that has no edges.)_

		make_pdb_data += "				tmp_cut_nodes_list.sort()\n"
		make_pdb_data += "				tmp_cut_nodes_list.reverse()\n"
		make_pdb_data += "				for tmp_cut_node_2 in range(len(tmp_cut_nodes_list)):\n"
		make_pdb_data += "					tmp_before_cut_node = nodes_for_select[tmp_cut_nodes_list[tmp_cut_node_2]]\n"
		make_pdb_data += "					for tmp_before_cut_edge_index in range(len(edges_for_select)):\n"
		make_pdb_data += "						if re.search(f'{tmp_before_cut_node}',str(edges_for_select[tmp_before_cut_edge_index])):\n"
		make_pdb_data += "							tmp_cut_edges_list.append(tmp_before_cut_edge_index)\n"		
		make_pdb_data += "					sub_mutations_data += f'{nodes_for_select[tmp_cut_nodes_list[tmp_cut_node_2]]},'\n"		
		make_pdb_data += "					del nodes_for_select[tmp_cut_nodes_list[tmp_cut_node_2]]\n"
		make_pdb_data += "					del values_for_rank[tmp_cut_nodes_list[tmp_cut_node_2]]\n"

		make_pdb_data += "				tmp_cut_edges_list_2 = list(set(tmp_cut_edges_list))\n"
		make_pdb_data += "				tmp_cut_edges_list_2.sort()\n"
		make_pdb_data += "				tmp_cut_edges_list_2.reverse()\n"
		make_pdb_data += "				for tmp_cut_edge in range(len(tmp_cut_edges_list_2)):\n"
		make_pdb_data += "					del edges_for_select[tmp_cut_edges_list_2[tmp_cut_edge]]\n"

		make_pdb_data += "				cut_graph_list.append('graph_before')\n"
		make_pdb_data += "				cut_graph_list[-1] = nx.Graph()\n"
		make_pdb_data += "				cut_graph_list[-1].add_nodes_from(nodes_for_select)\n"
		make_pdb_data += "				cut_graph_list[-1].add_edges_from(edges_for_select)\n"
		make_pdb_data += "				tmp_cut_nodes_list_2 = []\n"
		make_pdb_data += "				for tmp_isolated_node_index in range(len(nodes_for_select)):\n"
		make_pdb_data += "					if cut_graph_list[-1].degree(nodes_for_select[tmp_isolated_node_index]) == int(0):\n"
		make_pdb_data += "						tmp_cut_nodes_list_2.append(tmp_isolated_node_index)\n"
		make_pdb_data += "				tmp_cut_nodes_list_2.reverse()\n"
		make_pdb_data += "				for tmp_isolated_node in range(len(tmp_cut_nodes_list_2)):\n"
		make_pdb_data += "					sub_mutations_data += f'{nodes_for_select[tmp_cut_nodes_list_2[tmp_isolated_node]]},'\n"
		make_pdb_data += "					del nodes_for_select[tmp_cut_nodes_list_2[tmp_isolated_node]]\n"
		make_pdb_data += "					del values_for_rank[tmp_cut_nodes_list_2[tmp_isolated_node]]\n"

		make_pdb_data += "				fh_sub_mut_add_list = []\n"
		make_pdb_data += f"				fh_sub_mut_add_list.append('fh_sub_mut_add_{int(mutations_number_border)}')\n"
		make_pdb_data += f"				with open('sub_mutations_list_sel{mutations_number_tag}.fasta','a') as fh_sub_mut_add_list[-1]:\n"
		make_pdb_data += r"					fh_sub_mut_add_list[-1].write(f'{sub_mutations_data[:-1]}\n')"
		make_pdb_data += "\n"
		make_pdb_data += r"					fh_sub_mut_add_list[-1].write('beforeafter\n')"
		make_pdb_data += "\n				beforeafter_flag = int(1)\n"		

		# ___(If the nodes have not been deleted, these nodes are considered future candidates.)___

		make_pdb_data += "		selected_mutations = copy.deepcopy(nodes_for_select)\n"
		make_pdb_data += "		nodes_for_select_2 = copy.deepcopy(nodes_for_select)\n"
		make_pdb_data += "		if beforeafter_flag == int(0):\n"

		# _(Delete these nodes and associated edges, then also isolated nodes that has no edges.)_

		make_pdb_data += "			tmp_cut_nodes_list.sort()\n"
		make_pdb_data += "			tmp_cut_nodes_list.reverse()\n"
		make_pdb_data += "			for tmp_after_cut_node_index in range(len(tmp_cut_nodes_list)):\n"
		make_pdb_data += "				tmp_after_cut_node = nodes_for_select_2[tmp_cut_nodes_list[tmp_after_cut_node_index]]\n"
		make_pdb_data += "				for tmp_after_edge_index in range(len(edges_for_select)):\n"
		make_pdb_data += "					if re.search(f'{tmp_after_cut_node}',str(edges_for_select[tmp_after_edge_index])):\n"
		make_pdb_data += "						tmp_cut_edges_list.append(tmp_after_edge_index)\n"
		make_pdb_data += "				sub_mutations_data += f'{nodes_for_select_2[tmp_cut_nodes_list[tmp_after_cut_node_index]]},'\n"
		make_pdb_data += "				del nodes_for_select_2[tmp_cut_nodes_list[tmp_after_cut_node_index]]\n"
		make_pdb_data += "				del values_for_rank[tmp_cut_nodes_list[tmp_after_cut_node_index]]\n"

		make_pdb_data += "			tmp_cut_edges_list_2 = list(set(tmp_cut_edges_list))\n"
		make_pdb_data += "			tmp_cut_edges_list_2.sort()\n"
		make_pdb_data += "			tmp_cut_edges_list_2.reverse()\n"
		make_pdb_data += "			for tmp_after_cut_edge_index in range(len(tmp_cut_edges_list_2)):\n"
		make_pdb_data += "				del edges_for_select[tmp_cut_edges_list_2[tmp_after_cut_edge_index]]\n"

		make_pdb_data += "			cut_graph_list.append('graph_after')\n"
		make_pdb_data += "			cut_graph_list[-1] = nx.Graph()\n"
		make_pdb_data += "			cut_graph_list[-1].add_nodes_from(nodes_for_select_2)\n"
		make_pdb_data += "			cut_graph_list[-1].add_edges_from(edges_for_select)\n"
		make_pdb_data += "			tmp_after_cut_node_list_2 = []\n"
		make_pdb_data += "			for tmp_after_node_index in range(len(nodes_for_select_2)):\n"
		make_pdb_data += "				if cut_graph_list[-1].degree(nodes_for_select_2[tmp_after_node_index]) == int(0):\n"
		make_pdb_data += "					tmp_after_cut_node_list_2.append(tmp_after_node_index)\n"
		make_pdb_data += "			tmp_after_cut_node_list_2.reverse()\n"
		make_pdb_data += "			for tmp_after_isolated_node_index in range(len(tmp_after_cut_node_list_2)):\n"
		make_pdb_data += "				sub_mutations_data += f'{nodes_for_select_2[tmp_after_cut_node_list_2[tmp_after_isolated_node_index]]},'\n"
		make_pdb_data += "				del nodes_for_select_2[tmp_after_cut_node_list_2[tmp_after_isolated_node_index]]\n"
		make_pdb_data += "				del values_for_rank[tmp_after_cut_node_list_2[tmp_after_isolated_node_index]]\n"

		make_pdb_data += "			fh_sub_mut_add_list = []\n"
		make_pdb_data += f"			fh_sub_mut_add_list.append('fh_sub_mut_add_{int(mutations_number_border)}')\n"
		make_pdb_data += f"			with open('sub_mutations_list_sel{mutations_number_tag}.fasta','a') as fh_sub_mut_add_list[-1]:\n"
		make_pdb_data += r"				fh_sub_mut_add_list[-1].write('beforeafter\n')"
		make_pdb_data += "\n"
		make_pdb_data += r"				fh_sub_mut_add_list[-1].write(f'{sub_mutations_data[:-1]}\n')"
		make_pdb_data += "\n			sub_mutations_step += -1\n"

		# ___(Extract future candidates.)___

		make_pdb_data += "		for sub_step in range(sub_mutations_step):\n"
		make_pdb_data += "			if len(nodes_for_select_2) != int(0):\n"

		# _(Update the network and recalculate if the criterion is centrality.)_

		make_pdb_data += "				cut_graph_list.append(f'graph_after_{sub_step}')\n"
		make_pdb_data += "				cut_graph_list[-1] = nx.Graph()\n"
		make_pdb_data += "				cut_graph_list[-1].add_nodes_from(nodes_for_select_2)\n"
		make_pdb_data += "				cut_graph_list[-1].add_edges_from(edges_for_select)\n"

		if rank_flag == "centrality":

			if centrality_2_flag == "degree":
				make_pdb_data += "				centrality_values = list(nx.degree_centrality(cut_graph_list[-1]).values())\n"
			elif centrality_2_flag == "between":
				make_pdb_data += "				centrality_values = list(nx.betweenness_centrality(cut_graph_list[-1]).values())\n"
			elif centrality_2_flag == "close":
				make_pdb_data += "				centrality_values = list(nx.closeness_centrality(cut_graph_list[-1]).values())\n"

			make_pdb_data += "				values_for_rank = centrality_values\n"

		# _(Delete nodes with low values and associated edges, then also isolated nodes that has no edges.)_

		make_pdb_data += "				sub_mutations_data = ''\n"
		make_pdb_data += "				tmp_sub_step_cut_node = []\n"
		make_pdb_data += "				tmp_sub_step_cut_edge = []\n"
		make_pdb_data += "				tmp_sub_min_index = np.argmin(values_for_rank)\n"
		make_pdb_data += "				tmp_sub_min_value = values_for_rank[tmp_sub_min_index]\n"
		make_pdb_data += "				for tmp_sub_value in range(len(values_for_rank)):\n"
		make_pdb_data += "					if tmp_sub_min_value == values_for_rank[tmp_sub_value]:\n"
		make_pdb_data += "						tmp_sub_step_cut_node.append(tmp_sub_value)\n"
		make_pdb_data += "				tmp_sub_step_cut_node.sort()\n"
		make_pdb_data += "				tmp_sub_step_cut_node.reverse()\n"

		make_pdb_data += "				for tmp_sub_cut_node_index in range(len(tmp_sub_step_cut_node)):\n"
		make_pdb_data += "					tmp_sub_cut_node = nodes_for_select_2[tmp_sub_step_cut_node[tmp_sub_cut_node_index]]\n"
		make_pdb_data += "					for tmp_sub_edge_index in range(len(edges_for_select)):\n"
		make_pdb_data += "						if re.search(f'{tmp_sub_cut_node}',str(edges_for_select[tmp_sub_edge_index])):\n"
		make_pdb_data += "							tmp_sub_step_cut_edge.append(tmp_sub_edge_index)\n"
		make_pdb_data += "					sub_mutations_data += f'{nodes_for_select_2[tmp_sub_step_cut_node[tmp_sub_cut_node_index]]},'\n"
		make_pdb_data += "					del nodes_for_select_2[tmp_sub_step_cut_node[tmp_sub_cut_node_index]]\n"
		make_pdb_data += "					del values_for_rank[tmp_sub_step_cut_node[tmp_sub_cut_node_index]]\n"

		make_pdb_data += "				tmp_sub_step_cut_edge_2 = list(set(tmp_sub_step_cut_edge))\n"
		make_pdb_data += "				tmp_sub_step_cut_edge_2.sort()\n"
		make_pdb_data += "				tmp_sub_step_cut_edge_2.reverse()\n"
		make_pdb_data += "				for tmp_sub_cut_edge_index in range(len(tmp_sub_step_cut_edge_2)):\n"
		make_pdb_data += "					del edges_for_select[tmp_sub_step_cut_edge_2[tmp_sub_cut_edge_index]]\n"

		make_pdb_data += "				cut_graph_list.append(f'graph_after_2_{sub_step}')\n"
		make_pdb_data += "				cut_graph_list[-1] = nx.Graph()\n"
		make_pdb_data += "				cut_graph_list[-1].add_nodes_from(nodes_for_select_2)\n"
		make_pdb_data += "				cut_graph_list[-1].add_edges_from(edges_for_select)\n"
		make_pdb_data += "				tmp_sub_step_isolated_node = []\n"
		make_pdb_data += "				for tmp_sub_node_index in range(len(nodes_for_select_2)):\n"
		make_pdb_data += "					if cut_graph_list[-1].degree(nodes_for_select_2[tmp_sub_node_index]) == int(0):\n"
		make_pdb_data += "						tmp_sub_step_isolated_node.append(tmp_sub_node_index)\n"
		make_pdb_data += "				tmp_sub_step_isolated_node.reverse()\n"
		make_pdb_data += "				for tmp_sub_isolated_node_index in range(len(tmp_sub_step_isolated_node)):\n"
		make_pdb_data += "					sub_mutations_data += f'{nodes_for_select_2[tmp_sub_step_isolated_node[tmp_sub_isolated_node_index]]},'\n"
		make_pdb_data += "					del nodes_for_select_2[tmp_sub_step_isolated_node[tmp_sub_isolated_node_index]]\n"
		make_pdb_data += "					del values_for_rank[tmp_sub_step_isolated_node[tmp_sub_isolated_node_index]]\n"

		make_pdb_data += "				fh_sub_mut_add_list = []\n"
		make_pdb_data += f"				fh_sub_mut_add_list.append('fh_sub_mut_add_{int(mutations_number_border)}')\n"
		make_pdb_data += f"				with open('sub_mutations_list_sel{mutations_number_tag}.fasta','a') as fh_sub_mut_add_list[-1]:\n"
		make_pdb_data += r"					fh_sub_mut_add_list[-1].write(f'{sub_mutations_data[:-1]}\n')"
		make_pdb_data += "\n			else:\n"
		make_pdb_data += "				break\n"
		make_pdb_data += "		end_flag_for_nodecut = int(1)\n"

		# __________(Create a file of mutation adoption records.)__________

		make_pdb_data += f"with open('sub_mutations_list_sel{mutations_number_tag}.fasta','r') as fh_sub_mut_read_{int(mutations_number_border)}:\n"
		make_pdb_data += f"	sub_mutations_read = fh_sub_mut_read_{int(mutations_number_border)}.read()\n"
		make_pdb_data += r"sub_mutations_data = sub_mutations_read.split('\n')[:-1]"

		make_pdb_data += "\nsub_mut_forward_data = []\n"
		make_pdb_data += "for sub_data in range(len(sub_mutations_data)):\n"
		make_pdb_data += "	sub_mut_forward_data.append(sub_mutations_data[int('-'+f'{int(sub_data)+1}')])\n"
		make_pdb_data += r"sub_mut_forward_data_2 = '\n'.join(sub_mut_forward_data)"

		make_pdb_data += f"\nos.system('rm sub_mutations_list_sel{mutations_number_tag}.fasta')\n"
		make_pdb_data += f"with open('sub_mutations_list_sel{mutations_number_tag}.fasta','w') as fh_sub_mut_write_{int(mutations_number_border)}:\n"
		make_pdb_data += f"	fh_sub_mut_write_{int(mutations_number_border)}.write(sub_mut_forward_data_2)\n"
		
		return make_pdb_data
	# XXXXXXXXXXXXXXX(Not Recommended)XXXXXXXXXXXXXXX-(GOAL)

class FINAL_MUTATE():
# Create multiple PDBs by introducing mutations and determine the best variant in terms of score.

	def __init__(self,hisol_flag,stp_pose_from_pdb,stp_pose,sub_mutations_step,output_log,tag_set,mutations_number_border,mutations_number_tag):# <FITNESS_FUNCTION>

		# __________(Create a new directory to store the PDBs.)__________

		self.now_directory_list = os.listdir("./")
		if f"temporary_final_mutate_{tag_set}_sel{mutations_number_tag}" in self.now_directory_list:
			os.system(f"rm -f temporary_final_mutate_{tag_set}_sel{mutations_number_tag}/*")
		else:
			os.system(f"mkdir temporary_final_mutate_{tag_set}_sel{mutations_number_tag}")

		# __________(Extract mutational data and add ":".)__________

		fh_select_2_list = []
		fh_select_2_list.append(f'fh_select_2_{int(mutations_number_border)}')
		with open(f"select_mutations_{tag_set}_sel{mutations_number_tag}.fasta","r") as fh_select_2_list[-1]:
			self.select_mutations_data = fh_select_2_list[-1].read()

		self.select_mutations_data_2 = self.select_mutations_data[1:-1].replace("'","")
		self.select_mutations_list = self.select_mutations_data_2.split(", ")
		print(f"ORIGINAL MUTATIONS : {self.select_mutations_list}")

		for lmn in range(len(self.select_mutations_list)):
			self.select_mutations_list[lmn] = f"{self.select_mutations_list[lmn][:-1]}:{self.select_mutations_list[lmn][-1]}"

		# __________(Preparation of input data.)__________

		if hisol_flag == int(1):# <FITNESS_FUNCTION>-(START)
			self.O_selective_pressure_switch_O = "HISOL"
		else:
			self.O_selective_pressure_switch_O = "NOT-HISOL"# <FITNESS_FUNCTION>-(GOAL)

		with open("tmp_intmsa.out","r") as fh_intmsa:
			self.tmp_intmsa_data = fh_intmsa.read().split("\n")
		self.tmp_intmsa_data_2 = STANDARD_TOOLS.NULL_ELIMINATE(self.tmp_intmsa_data)
		del self.tmp_intmsa_data_2[0]
		
		self.SET_MUTATE(stp_pose_from_pdb,self.O_selective_pressure_switch_O,self.tmp_intmsa_data_2,stp_pose,self.select_mutations_list,sub_mutations_step,output_log,tag_set,mutations_number_border,mutations_number_tag)
	
	def SET_MUTATE(self,stp_pose_from_pdb,O_selective_pressure_switch_O,tmp_intmsa_data_2,stp_pose,select_mutations_list,sub_mutations_step,output_log,tag_set,mutations_number_border,mutations_number_tag):
	# Introduce selected mutations at a time.

		# __________(Extract variable mutational candidates for each step.)__________

		fh_sub_mutate_read_list = []
		fh_sub_mutate_read_list.append(f'fh_sub_mutate_read_{int(mutations_number_border)}')
		with open(f'sub_mutations_list_sel{mutations_number_tag}.fasta','r') as fh_sub_mutate_read_list[-1]:
			sub_mut_file_data = fh_sub_mutate_read_list[-1].read()
		sub_mut_file_list = sub_mut_file_data.split("\n")

		sub_mut_candidate = []
		beforeafter_check_flag = int(0)
		mutations_candidate_number = int(sub_mutations_step)
		sub_mutations_count = int(0)

		for sub_mut_data in range(len(sub_mut_file_list)):
			if sub_mutations_count == mutations_candidate_number:
				break

			if sub_mut_file_list[sub_mut_data] != 'beforeafter':
				sub_mut_candidate.append(sub_mut_file_list[sub_mut_data])
				if beforeafter_check_flag == int(1):
					sub_mutations_count += 1
			else:
				sub_mut_candidate.append(sub_mut_file_list[sub_mut_data])
				beforeafter_index = sub_mut_data
				beforeafter_check_flag = int(1)

		# __________(Use the mutational candidates to create lists of mutational groups for all patterns.)__________

		# _____(Remove unnecessary candidates and classify them as before or after.)_____

		if beforeafter_index >= int(mutations_candidate_number+1):
			del sub_mut_candidate[:beforeafter_index - mutations_candidate_number]
		print(f'FLEXIBLE MUTATIONAL CANDIDATES : {sub_mut_candidate}')
		
		before_sub_mut_candidate = []
		after_sub_mut_candidate = []
		for sub_mut_index in range(len(sub_mut_candidate)):
			if sub_mut_candidate[sub_mut_index] == "beforeafter":
				beforeafter_index_2 = sub_mut_index

		before_sub_mut_candidate = sub_mut_candidate[0:beforeafter_index_2]
		after_sub_mut_candidate = sub_mut_candidate[beforeafter_index_2+1:]
		before_sub_mut_candidate.reverse()
		print(f"BEFORE [REDUCE CANDIDATES] :{before_sub_mut_candidate}")
		print(f"AFTER  [ADD CANDIDATES]    :{after_sub_mut_candidate}")

		final_select_mutations_list_list = []
		final_select_mutations_list_list.append(select_mutations_list)# Original mutational pattern before changing the number of mutations.

		# _____(Produce patterns in which mutations are reduced from the original.)_____

		before_mutations = copy.deepcopy(select_mutations_list)
		for before_sub_mut in range(len(before_sub_mut_candidate)):

			if re.search(",",before_sub_mut_candidate[before_sub_mut]):
				before_sub_mut_data = before_sub_mut_candidate[before_sub_mut].split(",")
				for before_sub_mut_data_1 in range(len(before_sub_mut_data)):
					for exist_mut in range(len(before_mutations)):
						if f'{before_sub_mut_data[before_sub_mut_data_1][:-1]}:{before_sub_mut_data[before_sub_mut_data_1][-1]}' == before_mutations[exist_mut]:
							del before_mutations[exist_mut]
							break
			else:
				for exist_mut_1 in range(len(before_mutations)):
					if f'{before_sub_mut_candidate[before_sub_mut][:-1]}:{before_sub_mut_candidate[before_sub_mut][-1]}' == before_mutations[exist_mut_1]:
						del before_mutations[exist_mut_1]
						break

			if len(before_mutations) != int(0):				
				final_select_mutations_list_list.append(f"before_mutations_{before_sub_mut}")
				final_select_mutations_list_list[-1] = copy.deepcopy(before_mutations)
			else:
				break

		# _____(Produce patterns in which mutations are added from the original.)_____

		after_mutations = copy.deepcopy(select_mutations_list)
		for after_sub_mut in range(len(after_sub_mut_candidate)):

			if re.search(",",after_sub_mut_candidate[after_sub_mut]):
				after_sub_mut_data = after_sub_mut_candidate[after_sub_mut].split(",")
				for after_sub_mut_data_1 in range(len(after_sub_mut_data)):
					after_mutations.append(f"{after_sub_mut_data[after_sub_mut_data_1][:-1]}:{after_sub_mut_data[after_sub_mut_data_1][-1]}")
			else:
				after_mutations.append(f"{after_sub_mut_candidate[after_sub_mut][:-1]}:{after_sub_mut_candidate[after_sub_mut][-1]}")

			final_select_mutations_list_list.append(f"after_mutations_{after_sub_mut}")
			final_select_mutations_list_list[-1] = copy.deepcopy(after_mutations)

		# _____(Display all mutational patterns)_____

		print("SELECTED MUTATION'S PATTERNS :")
		for print_mut in range(len(final_select_mutations_list_list)):
			print(final_select_mutations_list_list[print_mut])

		# __________(Create PDBs with each mutational pattern.)__________

		# _____(Preparation of cycle.)_____

		reu_score_list = []# <FITNESS_FUNCTION>-(START)
		hisol_score_list = []# <FITNESS_FUNCTION>-(GOAL)
		final_pdb_list = []
		
		O_generation_number_switch_O = int(-10000)
		O_initial_flag_switch_O = int(1)
		O_library_hisol_score_O = int(0)
		X_X = int(1)
		X__X = str(2)

		# _____(Start cycle.)_____
		
		for select_mutations_pattern in range(len(final_select_mutations_list_list)):

			# ___(Introduced mutational pattern.)___

			tmp_select_mutations = final_select_mutations_list_list[select_mutations_pattern]
			print(f"TEMPORARY MUTATION'S PATTERN : {tmp_select_mutations}")

			final_task_pack_mut = []

			# ___(Mutagenesis.)___

			before_final_mutate_set_EM = CALCULATION_SCORE(stp_pose_from_pdb,final_task_pack_mut,O_generation_number_switch_O,O_selective_pressure_switch_O,O_initial_flag_switch_O,tmp_intmsa_data_2,stp_pose,O_library_hisol_score_O)# <FITNESS_FUNCTION>
			# Energy minimize.
			
			final_mutate_set = MUTATION(X_X,O_generation_number_switch_O,X_X,X_X,X_X,X_X,X__X,X__X,X__X,X__X,X__X,stp_pose_from_pdb,final_task_pack_mut,X__X,X_X,X__X,X__X,X__X,tmp_select_mutations,X_X,X_X,X_X)
			# Mutagenesis.

			# ___(Preparation of score calculation.)___
			
			final_mutation_pose = final_mutate_set.tmp_mutation_pose
			final_task_pack_mut = final_mutate_set.task_pack_mut

			if O_selective_pressure_switch_O == "HISOL":# <FITNESS_FUNCTION>-(START)
				O_initial_flag_switch_O = int(0)# For calculation of HiSol Score.
				if select_mutations_pattern == int(0):
					O_generation_number_switch_O = int(0)# For calculation of HiSol Score in libraries.<FITNESS_FUNCTION>-(GOAL)

			# ___(Score calculation.)___

			after_final_mutate_set_EM = CALCULATION_SCORE(final_mutation_pose,final_task_pack_mut,O_generation_number_switch_O,O_selective_pressure_switch_O,O_initial_flag_switch_O,tmp_intmsa_data_2,stp_pose,O_library_hisol_score_O)# <FITNESS_FUNCTION>
			# Energy minimize and calculate scores.
				
			final_reu_score = after_final_mutate_set_EM.reu_score# <FITNESS_FUNCTION>-(START)
			reu_score_list.append(final_reu_score)

			if O_selective_pressure_switch_O == "HISOL":

				if select_mutations_pattern == int(0):
					O_library_hisol_score_O = after_final_mutate_set_EM.library_hisol_score

				final_hisol_score = after_final_mutate_set_EM.hisol_score
				hisol_score_list.append(final_hisol_score)# <FITNESS_FUNCTION>-(GOAL)

			# ___(Create PDBs.)___

			if select_mutations_pattern == int(0):
				final_output_pdb_name = f"GAN-{tag_set}_sel{mutations_number_tag}_ORIGINAL.pdb"
			else:
				final_output_pdb_name = f"GAN-{tag_set}_sel{mutations_number_tag}_{select_mutations_pattern}.pdb"

			final_pdb_list.append(final_output_pdb_name)
			final_mutation_pose.dump_pdb(f"temporary_final_mutate_{tag_set}_sel{mutations_number_tag}/{final_output_pdb_name}")
			
			O_initial_flag_switch_O = int(1)
			O_generation_number_switch_O = int(-10000)

		# __________(Display scores of each PDB.)__________

		print(f"REU LIST         : {reu_score_list}")# <FITNESS_FUNCTION>-(START)
		if O_selective_pressure_switch_O == "HISOL":
			print(f"HiSol Score LIST : {hisol_score_list}")# <FITNESS_FUNCTION>-(GOAL)

		# __________(Evaluate the improvement of each PDB score by referring to the maximum and minimum scores in "output.log".)__________
		
		# _____(Read data from "output.log".)_____

		with open(output_log,"r") as fh_log_read:
			output_log_file_data = fh_log_read.read()
		output_log_generation_number = int(output_log_file_data.split("\n")[5].split(":")[-1])
		output_log_file_data_2 = output_log_file_data.split("\n")[16:]

		# _____(Extract data on elite scores for the first and last generations.)_____

		log_reu_score_list = []# <FITNESS_FUNCTION>-(START)
		log_hisol_score_list = []# <FITNESS_FUNCTION>-(GOAL)
		score_get_flag = int(0)
		pressure_check = ""
		for log_line in range(len(output_log_file_data_2)):

			if re.search("_",output_log_file_data_2[log_line]):
				score_get_flag = int(1)
				pressure_check = output_log_file_data_2[log_line].split("_")[-1]

			if (score_get_flag == int(1)) and (output_log_file_data_2[log_line][:1] == "0"):
				score_0gen = float(output_log_file_data_2[log_line].split(",")[-1])

				if pressure_check == "REU":# <FITNESS_FUNCTION>-(START)
					log_reu_score_list.append(score_0gen)
				elif (pressure_check == "HISOL") and (score_0gen != float(100000)):
					log_hisol_score_list.append(score_0gen)# <FITNESS_FUNCTION>-(GOAL)

			elif (score_get_flag == int(1)) and (re.search(":",output_log_file_data_2[log_line])):
				if int(output_log_file_data_2[log_line].split(":")[0]) == int(output_log_generation_number-1):		
					score_lastgen = float(output_log_file_data_2[log_line].split(",")[-1])				

					if pressure_check == "REU":# <FITNESS_FUNCTION>-(START)
						log_reu_score_list.append(score_lastgen)
					elif (pressure_check == "HISOL") and (score_lastgen != float(100000)):
						log_hisol_score_list.append(score_lastgen)# <FITNESS_FUNCTION>-(GOAL)

					score_get_flag = int(0)
					pressure_check = ""

		# _____(Calculate relative values from log data.)_____

		pressure_count = ""

		if len(log_reu_score_list) != int(0):# <FITNESS_FUNCTION>-(START)
			log_max_reu_score = np.max(log_reu_score_list)
			log_min_reu_score = np.min(log_reu_score_list)
			log_max_min_range = log_max_reu_score - log_min_reu_score

			reu_score_improvement_rate_list = []
			for tmp_reu_score in range(len(reu_score_list)):
				reu_score_improvement_rate = float(log_max_reu_score - reu_score_list[tmp_reu_score])/float(log_max_min_range)
				reu_score_improvement_rate_list.append(reu_score_improvement_rate)
			pressure_count += "REU"

		if len(log_hisol_score_list) != int(0):
			log_max_hisol_score = np.max(log_hisol_score_list)
			log_min_hisol_score = np.min(log_hisol_score_list)
			log_max_min_range_hisol = log_max_hisol_score - log_min_hisol_score
			if log_max_min_range_hisol == int(0):
				log_max_min_range_hisol = float(0.00001)

			hisol_score_improvement_rate_list = []
			for tmp_hisol_score in range(len(hisol_score_list)):
				if hisol_score_list[tmp_hisol_score] == 100000:
					hisol_score_improvement_rate_list.append(float(-100000))
				else:
					hisol_score_improvement_rate = float(log_max_hisol_score - hisol_score_list[tmp_hisol_score])/float(log_max_min_range_hisol)
					hisol_score_improvement_rate_list.append(hisol_score_improvement_rate)
			pressure_count += "HISOL"

		# __________(The best PDB is determined in terms of score improvement.)__________

		if pressure_count == "REU":
			score_improvement_rate_list = reu_score_improvement_rate_list

		elif pressure_count == "HISOL":
			score_improvement_rate_list = hisol_score_improvement_rate_list

		elif pressure_count == "REUHISOL":
			total_score_improvement_rate_list = []
			for rate_list_index in range(len(hisol_score_improvement_rate_list)):
				if hisol_score_improvement_rate_list[rate_list_index] == float(-100000):
					total_score_improvement_rate_list.append(float(-100000))
				else:
					total_score_improvement_rate = float(reu_score_improvement_rate_list[rate_list_index]) + float(hisol_score_improvement_rate_list[rate_list_index])# <FITNESS_FUNCTION>-(GOAL)
					total_score_improvement_rate_list.append(total_score_improvement_rate)
			score_improvement_rate_list = total_score_improvement_rate_list

		print(f"SCORE IMPROVE : {score_improvement_rate_list}")

		best_index = np.argmax(score_improvement_rate_list)
		best_mutations = final_select_mutations_list_list[best_index]

		for best_mutations_index in range(len(best_mutations)):
			best_mutations[best_mutations_index] = best_mutations[best_mutations_index].replace(":","")

		# _____(Complete residue number order or added mutations are not in residue number order.)_____

		# >>>>>>>>>>>>>>>VARIABLE<<<<<<<<<<<<<<<
		
		complete_residue_number_order_switch = "on"# In the case of "on", the output of mutations is in order of residue number.

		# ___(Sort by residue number.)___

		if complete_residue_number_order_switch == "on":
			tmp_order_best_mutations = []
			tmp_number_best_mutations = []
			for sss in range(len(best_mutations)):
				tmp_number_best = int(best_mutations[sss][:-1])
				tmp_number_best_mutations.append(tmp_number_best)

			while len(tmp_order_best_mutations) != len(tmp_number_best_mutations):
				tmp_min_number_best_index = np.argmin(tmp_number_best_mutations)
				tmp_min_number_mutation = best_mutations[tmp_min_number_best_index]
				tmp_order_best_mutations.append(tmp_min_number_mutation)
				tmp_number_best_mutations[tmp_min_number_best_index] = int(100000000)

			best_mutations = tmp_order_best_mutations

		# _____(Output of best mutations.)_____

		best_mutations_data = f"[{','.join(best_mutations)}]"

		fh_best_mut_list = []
		fh_best_mut_list.append(f'fh_best_mut_{int(mutations_number_border)}')
		with open(f"best_select_mutations_{tag_set}_sel{mutations_number_tag}.fasta","w") as fh_best_mut_list[-1]:
			fh_best_mut_list[-1].write(best_mutations_data)

		best_pdb_name = final_pdb_list[best_index]
		os.system(f"mv temporary_final_mutate_{tag_set}_sel{mutations_number_tag}/{best_pdb_name} temporary_final_mutate_{tag_set}_sel{mutations_number_tag}/{best_pdb_name.split('.')[0]}_BEST.pdb")

		# __________(Output of score.)__________

		best_reu_score = ""# <FITNESS_FUNCTION>-(START)
		best_hisol_score = ""
		original_reu_score = ""
		original_hisol_score = ""# <FITNESS_FUNCTION>-(GOAL)

		# _____(If the best pattern differs from the original.)_____

		if best_index != 0:

			if pressure_count == "REU":# <FITNESS_FUNCTION>-(START)
				best_reu_score = f"BEST REU     : {reu_score_list[best_index]}\n"
				original_reu_score = f"ORIGINAL REU : {reu_score_list[0]}"

			elif pressure_count == "HISOL":
				best_hisol_score = f"BEST HiSolScore     : {hisol_score_list[best_index]}\n"
				original_hisol_score = f"ORIGINAL HiSolScore : {hisol_score_list[0]}"

			elif pressure_count == "REUHISOL":
				best_reu_score = f"BEST REU            : {reu_score_list[best_index]}\n"
				best_hisol_score = f"BEST HiSolScore     : {hisol_score_list[best_index]}\n"
				original_reu_score = f"ORIGINAL REU        : {reu_score_list[0]}\n"
				original_hisol_score = f"ORIGINAL HiSolScore : {hisol_score_list[0]}"

		# _____(If the best pattern is the same as the original.)_____

		else:
			if pressure_count == "REU":
				best_reu_score = f"BEST / ORIGINAL REU : {reu_score_list[best_index]}"

			elif pressure_count == "HISOL":
				best_hisol_score = f"BEST / ORIGINAL HiSolScore : {hisol_score_list[best_index]}"

			elif pressure_count == "REUHISOL":
				best_reu_score = f"BEST / ORIGINAL REU        : {reu_score_list[best_index]}\n"
				best_hisol_score = f"BEST / ORIGINAL HiSolScore : {hisol_score_list[best_index]}"

		score_output_data = best_reu_score + best_hisol_score + original_reu_score + original_hisol_score# <FITNESS_FUNCTION>-(GOAL)

		fh_score_write_list = []
		fh_score_write_list.append(f'fh_score_write_{int(mutations_number_border)}')
		with open(f"pdb_score_{tag_set}_sel{mutations_number_tag}.fasta","w") as fh_score_write_list[-1]:
			fh_score_write_list[-1].write(score_output_data)

# ===============(Mutational selection main program.)===============

if change_GAOptimizer_flag == int(0):# When used as GAOptimizer.

	if skip_1_network_flag == int(0):# Extract mutational data and convert to network.

		# ==========(Extract mutational data from GAOptimizer results.)==========

		extract_mutation_data = EXTRACT_MUTATION_DATA(selective_pressure_list)

		mutations_list = extract_mutation_data.mutations_list

		print(f"MUTATIONAL DATA : {mutations_list}")

		# ==========(Create python file that construct a network using the extracted mutational data.)==========

		INITIAL_NETWORK(mutations_list,selective_pressure_list,reu_flag,hisol_flag,selective_pressure_variety)# <FITNESS_FUNCTION>

end_time_1 = time.time()
program_time_1 = end_time_1-start_time_1

# ===============(Create a file to calculate computation time.)==============

time_directory = os.listdir("./")
time_file_check = int(0)
for ttt in range(len(time_directory)):
	if time_directory[ttt] == "program_time.fasta":
		time_file_check = int(1)

if time_file_check == int(1):
	with open("program_time.fasta","a") as fh_time_1_a:
		fh_time_1_a.write(f"{program_time_1}\n")
elif time_file_check == int(0):
	with open("program_time.fasta","w") as fh_time_1_w:
		fh_time_1_w.write(f"{program_time_1}\n")

if change_GAOptimizer_flag == int(0):# When used as GAOptimizer.
	
	# ==========(Run the file.)==========
	
	os.system("python network_output.py")

	start_time_7 = time.time()

	# ==========(Reduce the network by deleting unchanged nodes.)==========

	network_cut = NETWORK_CUT(stp_residue_from_pdb,unchange_cut_flag)

	make_pdb_data = network_cut.make_pdb_data

	# ==========(Select original mutations and mutational candidates.)==========

	for border_index in range(len(mutations_number_border_list)):
		mutations_number_border = mutations_number_border_list[border_index]
		mutations_number_tag = mutations_number_tag_list[border_index]

		select_mutations = SELECT_MUTATIONS(make_pdb_data,rank_flag,select_flag,centrality_2_flag,extract_flag,extract_centrality_flag,step_flag,mutations_number_border,sub_mutations_step,tag_set,mutations_number_tag)

		make_pdb_data = select_mutations.make_pdb_data

	# =====(Create python file.)=====

	with open(f"extract_mutations_{tag_set}.py","w") as fh_extract:
		fh_extract.write(make_pdb_data)

	end_time_7 = time.time()
	program_time_7 = end_time_7-start_time_7
	with open("program_time.fasta","a") as fh_time_7:
		fh_time_7.write(f"{program_time_7}\n")

	# ==========(Run the file.)==========

	os.system(f"python extract_mutations_{tag_set}.py")

	start_time_6 = time.time()

	# ==========(Mutagenesis and determination of the best PDB.)==========

	fh_comparison_list = []
	for border_index_2 in range(len(mutations_number_border_list)):
		mutations_number_border = mutations_number_border_list[border_index_2]
		mutations_number_tag = mutations_number_tag_list[border_index_2]

		FINAL_MUTATE(hisol_flag,stp_pose_from_pdb,stp_pose,sub_mutations_step,output_log,tag_set,mutations_number_border,mutations_number_tag)# <FITNESS_FUNCTION>

		# ==========(Delete unchanged mutational data from STP.)==========

		mutation_file_list = ["best_select_mutations","select_mutations"]
		fh_list = [f"fh_1_best_{int(mutations_number_border)}",f"fh_2_select_{int(mutations_number_border)}"]
		fh_list_a = [f"fh_1_best_a_{int(mutations_number_border)}",f"fh_2_select_a_{int(mutations_number_border)}"]
		for mutation_file in range(len(mutation_file_list)):

			with open(f"{mutation_file_list[mutation_file]}_{tag_set}_sel{mutations_number_tag}.fasta","r") as fh_list[mutation_file]:
				mutation_list_file_read = fh_list[mutation_file].read()
			mutation_list_file_read_list = mutation_list_file_read[1:-1].split(",")

			delete_mutation_data_list = []
			for uuu in range(len(mutation_list_file_read_list)):
				now_mutation_data = mutation_list_file_read_list[uuu].strip()
				tmp_now_residue_number = int(now_mutation_data[:-1])
				tmp_after_residue = now_mutation_data[-1]
				tmp_before_residue = stp_residue_from_pdb[tmp_now_residue_number-1]
				if tmp_after_residue == tmp_before_residue:
					delete_mutation_data_list.append(uuu)
			delete_mutation_data_list.reverse()
			introduced_mutations = copy.deepcopy(mutation_list_file_read_list)
			for del_mut in range(len(delete_mutation_data_list)):
				del introduced_mutations[delete_mutation_data_list[del_mut]]

			introduced_mutations_data = ",".join(introduced_mutations)
			output_mutations_data = f"\nINTRODUCED MUTATIONS : [{introduced_mutations_data}]"
			with open(f"{mutation_file_list[mutation_file]}_{tag_set}_sel{mutations_number_tag}.fasta","a") as fh_list_a[mutation_file]:
				fh_list_a[mutation_file].write(output_mutations_data)
		
		fh_comparison_list.append(f"fh_comparison_{int(mutations_number_border)}")

		if border_index_2 >= 1:
			
			comparison_list = []
			for comparison_index in range(border_index_2-1, border_index_2+1):
				with open(f"best_select_mutations_{tag_set}_sel{mutations_number_tag_list[comparison_index]}.fasta","r") as fh_comparison_list[comparison_index]:
					tmp_comparison = fh_comparison_list[comparison_index].read()
				tmp_comparison_data = tmp_comparison.split("\n")[0]
				tmp_comparison_list = tmp_comparison_data[1:-1].split(",")
				comparison_list.append(len(tmp_comparison_list))

			if comparison_list[0] == comparison_list[1]:

				for comparison_index_2 in range(border_index_2-1, border_index_2+1):
					os.system(f"mv temporary_final_mutate_{tag_set}_sel{mutations_number_tag_list[comparison_index_2]} temporary_final_mutate_{tag_set}_sel{mutations_number_tag_list[comparison_index_2]}_CONVERGED")
					os.system(f"mv best_select_mutations_{tag_set}_sel{mutations_number_tag_list[comparison_index_2]}.fasta best_select_mutations_{tag_set}_sel{mutations_number_tag_list[comparison_index_2]}_CONVERGED.fasta")
					os.system(f"mv pdb_score_{tag_set}_sel{mutations_number_tag_list[comparison_index_2]}.fasta pdb_score_{tag_set}_sel{mutations_number_tag_list[comparison_index_2]}_CONVERGED.fasta")
				print("\nPDB creation, BEST determination, and score calculation are performed until the number of mutations converge !")
				break

	end_time_6 = time.time()
	program_time_6 = end_time_6-start_time_6
	with open("program_time.fasta","a") as fh_time_6:
		fh_time_6.write(f"{program_time_6}")
		
# ===============(Calculation of computation time.)===============

with open("program_time.fasta","r") as fh_time_r:
	program_time_data = fh_time_r.read()
program_time_list = program_time_data.split("\n")

total_program_time = float(0)
for jkl in range(len(program_time_list)):
	if (program_time_list[jkl] == "") or (program_time_list[jkl][:18] == "Total Program Time"):
		continue
	else:
		tmp_program_time = float(program_time_list[jkl])
		total_program_time += tmp_program_time

with open("program_time.fasta","a") as fh_time_a:
	fh_time_a.write(f"\n\nTotal Program Time : {total_program_time} seconds / {total_program_time/3600} hours\n")
print(f"\nTOTAL PROGRAM TIME : {total_program_time} seconds / {total_program_time/3600} hours")
