from flask import Flask, request, jsonify
import re
import spacy
import pyodbc
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np
from scipy import spatial
import codecs
import locationtagger

app = Flask(__name__)
ALLOWED_EXTENSIONS = ['[', ']', '{', '}', '(', ')', '+', '-', '/', '?', ',', ';', ':', '-', '_']


def clean(requirement):
    lis = re.split('\s', requirement)
    for symbol in ALLOWED_EXTENSIONS:
        if symbol in lis or symbol in requirement:
            return False


def length(requirement):
    lis = re.split('\s', requirement)
    if len(lis) > 15:
        return True


@app.route('/')
def index():
    return 'Hello welcome to login site. Try api'


@app.route('/Form', methods=['POST'])
def Form():
    if 'Question' not in request.values:
        noun_in_columnp = jsonify({'message': 'No file part in the request'})
        noun_in_columnp.status_code = 400
        return noun_in_columnp
    files = request.values.getlist('Question')
    global input_text
    input_text = ""

    for word in files:
        input_text += word

    errors = False
    success = False
    if clean(input_text) is None: success = True
    if length(input_text): errors = True

    if success and errors:
        response = jsonify({'message': 'Requirement is too long for the machine to understand'})
        response.status_code = 500
        return response
    if success:
        response = jsonify({'Success': 'Requirement is succesfully uploaded'})
        response.status_code = 201

        pd.set_option('display.max_columns', None)
        conn = pyodbc.connect('Driver={SQL Server};'
                              'Server=SSLTP11343\SQLEXPRESS;'
                              'Database=AH_ML;'
                              'Trusted_connection=Yes;')
        global noun_in_column, ext_noun
        # Load the NLP library
        spacy.load("en_core_web_sm")
        noun = []
        adjective = []
        verb = []
        numb = []
        ext_noun = ""
        noun_in_column = []
        added_place = []
        main_noun = ""
        main_verb = ""
        main_adj = ""
        number = 0
        noun_query = "SELECT Category,Sub_Category,Product_Name FROM stock_file"
        ddf = pd.read_sql(noun_query, conn)
        agg_set = {"max": ["Big", "Maximum", "Max", "High", "Top", "Great"],
                   "min": ["Small", "Minimum", "Min", "Low"]}
        column_agg_set = {"Items_sold": ["Sold", "Sell"], "Quantity": ["Quantity", "Value"],
                          "Discount": ["Discount", "Offer"], "Availability": ["Available", "Availability"]}
        comparison_set = {"<=": ["Lesserthan", "Lesser", "<"], ">=": ["Greaterthan", "Greater", ">"],
                          "=": ["Equalsto", "Equals", "Equal", "="]}

        def exists(list_, index_):
            try:
                if len(list_) > index_ >= 0:
                    return True
            except:
                return False

        def all_query_check(func_input_text):
            global all_data
            all_data = False
            print("in all chck")
            list_all = ["all", "every"]
            list_data = ["product", "products", "data"]
            input_list = re.split("\s", func_input_text)
            if len(input_list) == 1: input_list.insert(0, "all")

            print(input_list)
            for word in list_all:
                if word in input_list:
                    all_checking = True
                    print("inside all check")
                    all_index = input_list.index(word)
                    print(all_index)
                    if exists(input_list, (all_index + 1)) is True and input_list[(all_index + 1)] in list_data:
                        all_data = True
                    elif exists(input_list, (all_index - 1)) is True and input_list[(all_index - 1)] in list_data:
                        all_data = True
                    elif exists(input_list, (all_index + 2)) is True and input_list[(all_index - 2)] in list_data:
                        all_data = True
                    elif exists(input_list, (all_index - 2)) is True and input_list[(all_index + 2)] in list_data:
                        all_data = True
                    print(f"in all check {all_data}")
                    return all_checking

        def alternate_word(word_to_change):  # Load the pre-trained GloVe word embeddings
            embeddings_dict = {}
            with codecs.open(r"C:\My_Work\AH_ML\\trained\glove.6B.50d.txt", "r", encoding="utf-8") as f:
                for line in f:
                    values = line.split()
                    word_emd = values[0]
                    vector = np.asarray(values[1:], "float32")
                    embeddings_dict[word_emd] = vector

            def find_closest_embeddings(embedding):
                return sorted(embeddings_dict.keys(),
                              key=lambda word_embed: spatial.distance.euclidean(embeddings_dict[word_embed], embedding))

            return find_closest_embeddings(embeddings_dict[word_to_change])[:10]

        def column_connect(key, value):
            print(f"key : {key}")
            print(f"value : {value}")
            connect = " "
            if key != " ":
                connect = f" and {key} = '{value}'"
            return connect

        def noun_remover(lower_input_text):
            global ext_noun

            def clean_up(text_to_clean):
                print(f"before cleaning:{text_to_clean}")
                if "not" in text_to_clean:  # Cleaning stage 1
                    text_to_clean = text_to_clean.replace("not ", "not_")
                check_list = {"Out-Of-Stock": ["out of stock", "not_available"], "In-Stock": ["in stock", "available"],
                              "cell":["cell "]}
                for i, j in check_list.items():
                    for word in j:
                        if word in text_to_clean:
                            text_to_clean = text_to_clean.replace(word, i)
                print(f"After cleaning 1:{text_to_clean}")

                my_loc_list = ["seattle"]  # adding location not available in pretrained module
                text_token = word_tokenize(text_to_clean)
                for word in text_token:
                    if word.lower() in my_loc_list:
                        text_to_clean = text_to_clean.replace(word, "")
                        added_place.append(word.lower())

                text_token = word_tokenize(text_to_clean)  # Cleaning stage 2
                for word in text_token:
                    diction = {"phones": ["mobile", "mobilephone", "mobilephones"],
                               "Out-Of-Stock": ["not_available"], "In-Stock": ["available"],
                               "greater": ["above", "more"], "lesser": ["below", "under", "less"]}
                    for i, j in diction.items():
                        if word in j:
                            text_to_clean = re.sub(word, i, text_to_clean)
                print(f"After cleaning 2:{text_to_clean}")

                place_entity = locationtagger.find_locations(
                    text=text_to_clean)  # location check    # loading location pretrained module
                li_cou = place_entity.countries
                li_reg = place_entity.regions
                li_cities = place_entity.cities
                for word in li_cou, li_reg, li_cities:
                    if word != []:
                        place = word
                        for x in range(len(place)):
                            added_place.append(place[x])
                print(f"added_place list :{added_place}")
                if len(added_place) != 0:
                    for word in added_place:
                        text_to_clean = text_to_clean.replace(word.lower(), "")
                print(f"After cleaning 3:{text_to_clean}")

                return text_to_clean

            cleaned_text = clean_up(lower_input_text)
            print(f"cleaned_text :{cleaned_text}")
            text_tokens = word_tokenize(cleaned_text)  # NLP - Stopword to remove unwanted words
            tokens_without_sw = [word for word in text_tokens if not word in stopwords.words()]
            texted = ' '.join(tokens_without_sw)
            text_tokens = word_tokenize(texted)
            print(f"removed stopword :{text_tokens}")

            # Noun removing and finding noun column-name
            modified_text = texted
            li = []
            try:
                for token in text_tokens:
                    alternate_list = alternate_word(token)
                    for alt_word in alternate_list:
                        column_with_value = ddf.columns[(ddf == alt_word.title()).any()].tolist()
                        if column_with_value:
                            noun_in_column.append(column_with_value[0])
                            li.append(alt_word.title())
                            ext_noun = li[0]
                            modified_text = texted.replace(token, "")
                if noun_in_column:
                    print(f"colum_name_noun_in_column: {noun_in_column}")
                    print(f"ext_noun: {ext_noun}")
                    return modified_text
                else:
                    noun_in_column.append(" ")
                    print(f"colum_name_noun_in_column: {noun_in_column}")
                    print(f"ext_noun: {ext_noun}")
                    print(f"modified_text :{modified_text}")
                    return modified_text

            except:
                if noun_in_column:
                    print(f"colum_name_noun_in_column: {noun_in_column}")
                    print(f"ext_noun: {ext_noun}")
                    return modified_text
                else:
                    noun_in_column.append(" ")
                    print(f"colum_name_noun_in_column: {noun_in_column}")
                    print(f"ext_noun: {ext_noun}")
                    return modified_text

        def find_any_column(value):
            print("---------")
            print(value)
            empty = " "
            if value == 'available' or value == 'not_available':
                return 'Availability'
            else:
                overall_query = "SELECT * FROM stock_file"
                dft = pd.read_sql(overall_query, conn)
                column_with_value = dft.columns[(dft == value.title()).any()].tolist()
                if column_with_value:
                    return column_with_value[0]
                else:
                    return empty

        def location_exists(place):
            column_name2 = []
            location_final = ""
            if place:
                for place_word in place:
                    column_name2.append(find_any_column(place_word))
                    print(column_name2)
                print(f"added_place :{place}")
                print(f"column_name2 :{column_name2}")
                if len(place) == len(column_name2):
                    for x, y in zip(column_name2, place):
                        y = "'" + y + "'"
                        # location_dict = dict(zip(listed_place, column_name2))
                        if x not in location_final:
                            location_final += ' and ' + x + " = " + y
                        elif x in location_final:
                            location_final += ' or ' + x + " = " + y
                    return location_final

        def pos(texted):
            print(texted)
            tokens = word_tokenize(texted)
            tags = nltk.pos_tag(tokens, tagset="universal")
            noun.append(ext_noun)
            print(f"Pos:{tags}")
            for token in tags:
                if token[1] == "NOUN" or token[1] == "ADJ":
                    adjective.append(token[0])
                elif token[1] == "VERB":
                    verb.append(token[0])
                elif token[1] == "NUM":
                    numb.append(token[0])
            return tags

        def check(list_text):
            x_list = []
            print(f"list_text: {list_text}")
            if not list_text:
                return 3

            elif len(list_text) == 1:
                if number == list_text[0] :
                    list_text.insert(0, "price")
            dic_set = {
                0: ["Lesserthan", "Lesser", "<", "Greaterthan", "Greater", ">", "Equalsto", "Equals", "Equal", "=",
                    "above", "in",
                    "below"],
                1: ["Big", "Maximum", "Max", "High", "Top", "Great", "Small", "Minimum", "Min", "Low"],
                2: ['Out-Of-Stock', 'In-Stock', 'Available', 'Not_Available']}
            for x, y in dic_set.items():
                for word in list_text:
                    if word.title() in y:
                        x_list.append(x)
                        return x_list[0]

        def find_comparison_column(vrb, adj_list, numb, added_comp):
            print(adj_list)
            print(type(adj_list))
            comparison = ""
            if len(numb) == 1:
                for adj in adj_list:
                    for i, j in comparison_set.items():
                        if adj.title() in j:
                            comparison = i
                        elif vrb.title() in j:
                            comparison = i
                        elif added_comp.title() in j:
                            comparison = i
                return comparison
            elif len(numb) > 1:
                def exists(x):
                    try:
                        if len(text_list) > x >= 0:
                            return True
                    except:
                        return False

                comp_value = ""
                text_list = re.split(" ", input_text)
                comp_list_above = ["above", "greater"]
                comp_list_below = ["below", "lesser"]
                for numer in numb:
                    val = text_list.index(numer)
                    if exists(val - 1) is True and text_list[val - 1] in comp_list_above:
                        comp_value += ">" + numer
                    elif exists(val + 1) is True and text_list[val + 1] in comp_list_above:
                        comp_value += ">" + numer
                    elif exists(val - 2) is True and text_list[val - 2] in comp_list_above:
                        comp_value += ">" + numer
                    elif exists(val + 2) is True and text_list[val + 2] in comp_list_above:
                        comp_value += ">" + numer
                    elif exists(val - 1) is True and text_list[val - 1] in comp_list_below:
                        comp_value += "<" + numer
                    elif exists(val + 1) is True and text_list[val + 1] in comp_list_below:
                        comp_value += "<" + numer
                    elif exists(val - 2) is True and text_list[val - 2] in comp_list_below:
                        comp_value += "<" + numer
                    elif exists(val + 2) is True and text_list[val + 2] in comp_list_below:
                        comp_value += "<" + numer
                return comp_value

        def find_agg_column(m_verb):
            agg_col_name = ""
            for i, j in column_agg_set.items():
                if m_verb in j:
                    agg_col_name = i
            return agg_col_name

        def find_aggregation(m_adj):
            aggregation = ""
            for i, j in agg_set.items():
                if m_adj in j:
                    aggregation = i
            return aggregation

        def comparison_query(column_value, comp_column, comparison_value, numb, locat_final, add_on):
            print(numb)
            if len(numb) == 1:
                print("in comp1")
                # select category,sub_category,product_name,price,availability from stock_file where sub_category ='Chairs' and price>500
                query = '''select category,sub_category,product_name,{} price,availability from stock_file where 1=1 {} and {} 
                {} {} {}'''.format(
                    add_on, column_value, comp_column, comparison_value, number, locat_final)
                output_query = pd.read_sql(query, conn)
                print(query)
                return output_query
            elif len(numb) > 1:
                print("in comp2")
                index_of_sign = comparison_value.index(re.findall("[<>]", comparison_value)[1])
                comp1 = comparison_value[:index_of_sign] + ' and '
                comp2 = comparison_value[index_of_sign:]
                # select category,sub_category,product_name,price,availability from stock_file where sub_category ='Chairs' and price>500
                query = '''select category,sub_category,product_name,{} price,availability from stock_file where 1=1 {} and {} 
                        {} {} {} {}'''.format(
                    add_on, column_value, comp_column, comp1, comp_column, comp2, locat_final)
                output_query = pd.read_sql(query, conn)
                print(query)
                return output_query

        def max_query(m_aggregation, m_column_agg_value, m_column_value, locate, add_on):
            if numb:
                query = '''select top {} Sub_Category, Product_Name,{} Price, {}({}) from stock_file
                                where 1=1 {} {} group by Sub_Category, Product_Name,{} Price order by {}({}) desc''' \
                    .format(numb[0], add_on, m_aggregation, m_column_agg_value, m_column_value, locate,
                            add_on,
                            m_aggregation,
                            m_column_agg_value)
                print(query)
                output_query = pd.read_sql(query, conn)
            else:
                query = '''select Sub_Category, Product_Name,{} Price, {}({}) from stock_file
                    where 1=1 {} {} group by Sub_Category, Product_Name,{} Price order by {}({}) desc''' \
                    .format(add_on, m_aggregation, m_column_agg_value, m_column_value, locate, add_on,
                            m_aggregation, m_column_agg_value)
                print(query)
                output_query = pd.read_sql(query, conn)
            return output_query

        def available_query(column_value, column_name_av, locat_final, add_on):
            query = '''select category, sub_category, product_name,{} price, availability from stock_file where 1=1 {} {} {} ''' \
                .format(add_on, column_value, column_name_av, locat_final)
            print(query)
            output_query = pd.read_sql(query, conn)
            return output_query

        def fetchall_query(noun_column_q, final_place_q):
            query = '''select * from  stock_file  where 1=1 {} {}'''.format(noun_column_q, final_place_q)
            print(query)
            output_query = pd.read_sql(query, conn)
            return (output_query)
        try:
            input_text = input_text.lower()  # switching to lowercase
            all_check = all_query_check(input_text)
            print(f"all_data_for no noun {all_data}")
            if all_check:
                if all_data is True:
                    print("inside all if")
                    noun_remover(input_text)
                    final_place = location_exists(added_place)
                    if final_place is None:
                        final_place = ""
                    output0 = fetchall_query("", final_place)
                    print(output0)
                    print(output0.to_json(orient='records'))
                    return output0.to_json(orient='records')

                else:
                    print("in all else")
                    noun_remover(input_text)
                    print(f"extracted_noun:{ext_noun}")
                    print(f"column_name_for_noun:{noun_in_column[0]}")
                    noun_column = column_connect(noun_in_column[0], ext_noun)
                    final_place = location_exists(added_place)
                    if final_place is None:
                        final_place = ""
                    print(noun_column)
                    print(final_place)
                    output01 = fetchall_query(noun_column, final_place)
                    print(output01)
                    print(output01.to_json(orient='records'))
                    return output01.to_json(orient='records')

            else:
                text_without_noun = noun_remover(input_text)
                print(f"text_without_noun:{text_without_noun}")
                no_noun_list = re.split("\s", text_without_noun)
                no_noun_list = [ele for ele in no_noun_list if ele != '']  # List without noun
                print(f"text_without_noun:{text_without_noun}")
                print(f"extracted_noun:{ext_noun}")
                print(f"column_name_for_noun:{noun_in_column[0]}")
                pos(text_without_noun)
                if len(noun) != 0:
                    main_noun = noun[len(noun) - 1]
                if len(verb) != 0:
                    main_verb = verb[len(verb) - 1].title()
                if len(adjective) != 0:
                    main_adj = adjective[len(adjective) - 1].title()
                if len(numb) != 0:
                    number = numb[0]

                opt = True
                for i, j in comparison_set.items():
                    for sym in j:
                        if sym.lower() in no_noun_list:
                            opt = False
                added_comp = ""
                if (main_adj.lower() in no_noun_list or main_noun) and number in no_noun_list and opt:
                    print("in added comp")
                    if main_adj.lower() in no_noun_list:
                        adj_index = no_noun_list.index(main_adj.lower())
                    elif main_noun:
                        adj_index = 1
                    numb_index = no_noun_list.index(number)
                    if 2 > numb_index - adj_index >= -1 and numb_index - adj_index != 0:
                        no_noun_list.insert(max(numb_index, adj_index), '=')
                        added_comp = '='

                print(adjective)
                print(f"noun:{noun}")
                print(f"main_noun : {main_noun}")
                print(f"main_verb : {main_verb}")
                print(f"main_adj : {main_adj}")
                print(f"numb : {numb}")
                print(f"adj:{adjective}")
                column_value = noun_in_column[0]
                print(f"number : {number}")
                query_value = check(no_noun_list)
                print(f"query_value :{query_value}")
                print("------------")

                if query_value == 0:
                    column_name2 = []
                    add_on = ""
                    comparison_value = find_comparison_column(main_verb, adjective, numb, added_comp)
                    print(comparison_value)
                    comp_column = "Price"
                    print("in comp")
                    print(f"comp_column :{comp_column}")
                    print(f"column_value :{column_value}")
                    print(f"Comparison operator :{comparison_value}")
                    print(f"number  :{number}")
                    print(f"noun_value :{main_noun}")
                    noun_comp_column = column_connect(column_value, main_noun)
                    if noun_comp_column is None:
                        noun_comp_column = ""
                    print(f"noun_column : {noun_comp_column}")
                    final_place = location_exists(added_place)
                    if final_place is None:
                        final_place = ""
                    for x in list(set(column_name2)):
                        add_on += x + " , "
                    print(f"final_place:{final_place}")
                    output = comparison_query(noun_comp_column, comp_column, comparison_value, numb, final_place,
                                              add_on)
                    print(output)
                    return output.to_json(orient='records')

                elif query_value == 1:
                    add_on = ""
                    print("inside top")
                    print(f"column_value :{column_value}")
                    column_agg_value = find_agg_column(main_verb)
                    print(f"column_agg :{column_agg_value}")
                    aggregation = find_aggregation(main_adj)
                    print(f"aggregation :{aggregation}")
                    print(f"noun_value :{main_noun}")
                    column_name2 = []
                    noun_agg_column = column_connect(column_value, main_noun)
                    if noun_agg_column is None:
                        noun_agg_column = ""
                    print(f"noun_agg_column : {noun_agg_column}")
                    final_place = location_exists(added_place)
                    if final_place is None:
                        final_place = ""
                    for x in list(set(column_name2)):
                        add_on += x + " , "
                    print(f"final_place:{final_place}")
                    output2 = max_query(aggregation, column_agg_value, noun_agg_column, final_place, add_on)
                    print(output2)
                    return output2.to_json(orient='records')


                elif query_value == 2:
                    column_name2 = []
                    add_on = ""
                    print("in stock")
                    print(f"main_noun :{main_noun}")
                    print(f"main_adj :{main_adj}")
                    print(f"column_value :{column_value}")
                    column_name_av = find_any_column(main_adj)
                    print(f"column_name_av :{column_name_av}")
                    noun_column = column_connect(column_value, main_noun)
                    if noun_column is None:
                        noun_column = ""
                    print(f"noun_column : {noun_column}")
                    adj_column = column_connect(column_name_av, main_adj)
                    if adj_column is None:
                        adj_column = ""
                    print(f"adj_column : {adj_column}")
                    final_place = location_exists(added_place)
                    if final_place is None:
                        final_place = ""
                    for x in list(set(column_name2)):
                        add_on += x + " , "
                    print(f"final_place:{final_place}")
                    output3 = available_query(noun_column, adj_column, final_place, add_on)
                    print(output3)
                    return output3.to_json(orient='records')

                elif query_value == 3:
                    print(f"extracted_noun:{ext_noun}")
                    print(f"column_name_for_noun:{noun_in_column[0]}")
                    noun_column = column_connect(noun_in_column[0], ext_noun)
                    print(f"added_place :{added_place}")
                    final_place = location_exists(added_place)
                    if final_place is None:
                        final_place = ""
                    print(noun_column)
                    print(f"final_place :{final_place}")
                    output04 = fetchall_query(noun_column, final_place)
                    print(output04)
                    print(output04.to_json(orient='records'))
                    return output04.to_json(orient='records')
                print("the end")
        except:
            response = jsonify(
                {'Error': 'Sorry unable to read your question'})
            response.status_code = 500
            return response
    else:
        print("last")
        response = jsonify(
            {'Error': 'Sorry unable to read your requirement.Remove signs or check for the error and try again'})
        response.status_code = 500
        return response


if __name__ == "__main__":
    app.run(debug=True)
