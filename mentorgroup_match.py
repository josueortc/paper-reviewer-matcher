#!/usr/bin/env python

"""MindMatch: a script for matching people to people in the conference
Run the script here since

Usage:
  mindmatch.py PATH [--n_match=<n_match>] [--n_trim=<n_trim>] [--output=<output>]
  mindmatch.py [-h | --help]
  mindmatch.py [-v | --version]

Arguments:
  PATH                  Path to a CSV file,
                        a file need to have ('user_id', 'fullname', 'abstracts', 'conflicts') in the header

Options:
  -h --help             Show documentation helps
  --version             Show version
  --n_match=<n_match>   Number of match per user
  --n_trim=<n_trim>     Trimming parameter for distance matrix, increase to reduce problem size
  --output=<output>     Output CSV file contains 'user_id' and 'match_ids' which has match ids with ; separated
"""

import os
import sys

import numpy as np
import pandas as pd
from docopt import docopt
import time
from ortools.linear_solver import pywraplp
from paper_reviewer_matcher import preprocess, affinity_computation, create_lp_matrix, create_assignment, affinity_time
from fuzzywuzzy import fuzz
from tqdm import tqdm


def linprog(f, A, b):
    """
    Solve the following linear programming problem
            maximize_x (f.T).dot(x)
            subject to A.dot(x) <= b
    where   A is a sparse matrix (coo_matrix)
            f is column vector of cost function associated with variable
            b is column vector
    """

    # flatten the variable
    f = f.ravel()
    b = b.ravel()

    solver = pywraplp.Solver('SolveReviewerAssignment',
                             pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)

    infinity = solver.Infinity()
    n, m = A.shape
    x = [[]] * m
    c = [0] * n

    for j in range(m):
        x[j] = solver.NumVar(-infinity, infinity, 'x_%u' % j)

    # state objective function
    objective = solver.Objective()
    for j in range(m):
        objective.SetCoefficient(x[j], f[j])
    objective.SetMaximization()

    # state the constraints
    for i in range(n):
        c[i] = solver.Constraint(-infinity, int(b[i]))
        for j in A.col[A.row == i]:
            c[i].SetCoefficient(x[j], A.data[np.logical_and(A.row == i, A.col == j)][0])

    result_status = solver.Solve()
    if result_status != 0:
        print("The final solution might not converged")

    x_sol = np.array([x_tmp.SolutionValue() for x_tmp in x])

    return {'x': x_sol, 'status': result_status}


def compute_conflicts(df):
    """
    Compute conflict for a given dataframe
    """
    cois = []
    for i, r in tqdm(df.iterrows()):
        exclude_list = r['conflicts'].split(';')
        for j, r_ in df.iterrows():
            if max([fuzz.ratio(r_['fullname'], n) for n in exclude_list]) >= 85:
                cois.append([i, j])
                cois.append([j, i])
    return cois


if __name__ == "__main__":
    arguments = docopt(__doc__, version='MindMatch 0.1.dev')
    check = time.time()
    file_name = arguments['PATH']
    df = pd.read_pickle(file_name)
    print("Number of people in the file = {}".format(len(df)))

    n_match = arguments.get('--n_match')
    if n_match is None:
        n_match = 1
        print('<n_match> is set to default for 6 match per user')
    else:
        n_match = int(n_match)
        print('Number of match is set to {}'.format(n_match))
    assert n_match >= 1, "You should set <n_match> to be more than 2"

    n_trim = arguments.get('--n_trim')
    if n_trim is None:
        n_trim = 0
        print('<n_trim> is set to default, this will take very long to converge for a large problem')
    else:
        n_trim = int(n_trim)
        print('Trimming parameter is set to {}'.format(n_trim))

    output_filename = arguments.get('output')
    if output_filename is None:
        output_filename = 'output_match_mentor_student.csv'

    # create assignment matrix
    student = df[df['Type'] == 'Student']
    data_student = student['Availability']
    data_s = [data_student[i] for i in range(len(data_student))]
    data_student = np.stack(data_s, axis=0)
    mentor = df[df['Type'] == 'Mentor']
    data_mentor = mentor['Availability']
    data_m = [data_mentor[len(data_student)+i] for i in range(len(data_mentor))]
    data_mentor = np.stack(data_m, axis=0)
    if n_trim != 0:
        data_mentor = data_mentor[:n_trim,:]
        data_student = data_student[:n_trim,:]

    A_trim = affinity_time(data_student, data_mentor)


    print('Compute conflicts... (this may take a bit)')
    print('Done computing conflicts!')

    print('Finishe with A', time.time() - check)
    print('Solving a matching problem...')
    v, K, d = create_lp_matrix(A_trim,
                               min_reviewers_per_paper=n_match, max_reviewers_per_paper=n_match,
                               min_papers_per_reviewer=n_match, max_papers_per_reviewer=n_match)
    x_sol = linprog(v, K, d)['x']
    b = create_assignment(x_sol, A_trim)
    if (b.sum() == 0):
        print('Seems like the problem does not converge, try reducing <n_trim> but not too low!')
    else:
        print('Successfully assigned all the match!')

    if (b.sum() != 0):
        output = []
        user_ids_map_student = {ri: r['hash'] for ri, r in df[df['Type'] == 'Student'].iterrows()}
        user_ids_map_mentor = {ri: r['hash'] for ri, r in df[df['Type'] == 'Mentor'].iterrows()}
        for i in range(len(b)):
            match_ids = [str(user_ids_map_mentor[len(data_s) + b_]) for b_ in np.nonzero(b[i])[0]]
            output.append({
                'user_id': user_ids_map_student[i],
                'match_ids': ';'.join(match_ids)
            })
        output_df = pd.DataFrame(output)
        output_df.to_csv(output_filename, index=False)
        print('Successfully save the output match to {}'.format(output_filename))
    print('Finish all', time.time() - check)
