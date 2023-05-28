
        try:
            parsedResults.append("Data: " + format(answerbox))
            tempSourcesMemory[0].append("Google search: " + search_res["search_metadata"]["google_url"])
        except:
            print("Error, could not add data from answerbox: " + answerbox)
