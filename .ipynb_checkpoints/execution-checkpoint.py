def grid_search(args,combinations):
    if args.verbose:
        print("Testing {} combinations in total".format(sum([len(i) for i in combinations.values()]))
    for task_param in combinations:
        tester = Tester(task_param,task_name,task_model,task_data)
        tester.run()
        tester.log(f"./results/{args.task_name}_gridsearch.log")
    # save results
    with open("./results/results_gridsearch.pickle", "wb") as f:
        pickle.dump(results, f)    