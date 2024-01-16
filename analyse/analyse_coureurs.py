import matplotlib.pyplot as plt
import pandas as pd


def process_data_and_plot(id_coureur):
    datetimes = []
    athletes = []
    distances = []
    durations = []
    genders = []
    age_groups = []
    countries = []
    majors = []

    # données de volume quotidien
    df = pd.read_parquet('../../data/run_ww_2019_d.parquet')
    df2 = pd.read_parquet('../../data/run_ww_2020_d.parquet')

    # coureur semi pro, 18 - 34
    coureur_2019 = df.loc[df.athlete == id_coureur]
    coureur_2020 = df2.loc[df2.athlete == id_coureur]

    dataframes = [coureur_2019, coureur_2020]

    for data in dataframes:
        for index, row in data.iterrows():
            datetimes.append(row['datetime'])
            athletes.append(row['athlete'])
            distances.append(row['distance'])
            durations.append(row['duration'])
            genders.append(row['gender'])
            age_groups.append(row['age_group'])
            countries.append(row['country'])
            majors.append(row['major'])
    count = 0

    for i in distances:
        if i == 0:
            count += 1

    print("--------------------------------------------")
    print("jours de repos dans l'année : " + str(count))
    print("moyenne de nombre de jour par semaine : " + str(count / 54))
    print("gender : " + str(genders[0]))
    print("age_groups : " + str(age_groups[0]))
    print("countries : " + str(countries[0]))
    print("majors : " + str(majors[0]))
    print("volume total : " + str(sum(distances)))
    print("--------------------------------------------")
    plt.figure(figsize=(12, 8))

    plt.scatter(datetimes, distances, label="datetime / distance", s=100)

    plt.title('Analyse entrainement : volume quotidien en fonction de la date sur 2 ans')
    plt.xlabel('Date')
    plt.ylabel('Distance (km)')

    plt.show()

    plt.scatter(durations, distances, label="duration / distance", s=100)

    plt.title('Analyse entrainements : durée en fonction de la distance sur 2 ans')
    plt.xlabel('Durée (minutes)')
    plt.ylabel('Distance (km)')

    plt.show()


# coureur semi pro, 18 - 34
process_data_and_plot(37594)

# coureur récréatif, 18 - 34
process_data_and_plot(21550)

# autres profils intéressants à observer
process_data_and_plot(21547)
process_data_and_plot(21549)
process_data_and_plot(15468)
process_data_and_plot(13645)
process_data_and_plot(19847)
process_data_and_plot(15264)
process_data_and_plot(21551)



