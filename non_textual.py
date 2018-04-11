
import util


def main():
    X,y = util.load_reaction_counts('merged.csv')
    print X
    print '------------------------'
    print y
    ################################################################
    ## Predict factuality from number of                          ##
    ## shares, comments, likes, loves, wows,hahas, sads, angrys   ##
    ################################################################
    # -- This is about if people respond to posts depending on    ## 
    # -- the factuality of the post. (this might be some          ##
    # -- underlying cusation of media intention)                  ##
    ################################################################
    



if __name__ == "__main__" :
    main()