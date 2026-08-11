### Worst 20 translations — en-es

| # | sBLEU | Source | Reference | Model output | Failure modes |
|---|-------|--------|-----------|--------------|---------------|
| 1 | 0.0 | Tell me about it! | Ni me lo digas. | ¡Cuéntame sobre ello! | no_content_overlap |
| 2 | 0.0 | Sit down! | ¡Sentaos! | Siéntese. | no_content_overlap |
| 3 | 0.0 | Curse you! | Te maldigo. | ¡Durmentaos! | no_content_overlap |
| 4 | 0.0 | That's really great! | Es realmente magnífico. | ¡Qué bien es eso! | no_content_overlap |
| 5 | 0.0 | Sit down! | ¡Sentate! | Siéntese. | no_content_overlap |
| 6 | 0.0 | Tell me about it. | ¡Dímelo a mí! | Háblame de ello. | no_content_overlap |
| 7 | 0.0 | Sit down! | ¡Siéntate! | Siéntese. | no_content_overlap |
| 8 | 3.0 | Wealthy older men often marry younger trophy wives. | Hombres ricos suelen casarse con jóvenes mujeres trofeo. | A menudo, los hombres mayores, a menudo nos casan con las trofeas más jóvenes. | over_generation |
| 9 | 3.4 | A strong argument for the religion of Christ is this - that offences against Charity are about the only ones which men on their death-beds can be made, not to understand, but to feel, as crime. | Un sólido argumento en favor del cristianismo es el siguiente: las ofensas contra la caridad es probablemente lo único que, en sus lechos de muerte, los hombres llegan a sentir y no a comprender como un crimen. | Un argumento fuerte para la religión de Cristo es esta - que las oficiales de Charualidad están a punto de los únicos que los hombres de su muerte pueden ser obligados, no para entender, sino que no se sienten el crimen. | other |
| 10 | 3.5 | The squadron encountered an ambush and scrambled for coverage. | El escuadrón se topó con una emboscada y se apresuró a ponerse a cubierto. | El caspa encontró un embosco de ambush y scorrocaba por cobertura. | no_content_overlap |
| 11 | 3.6 | His meaning is quite plain. | Lo que él trata de decir es bien sencillo de comprender. | Su significado es bastante simple. | truncation, no_content_overlap |
| 12 | 3.7 | Peasants often have a secondary activity to augment their income. | A menudo los campesinos se ocupan en una actividad secundaria para complementar sus ingresos. | Los Peas a menudo tienen una segunda actividad mayor a un aumento de sueldo. | other |
| 13 | 3.8 | Mr. Smith makes it a rule to take a walk every morning. | El Señor Smith tiene el hábito de salir a caminar todas las mañanas. | El señor Smith hace como norma dar un paseo cada mañana. | other |
| 14 | 4.0 | You can go out on condition that you come home by seven. | Si vuelves antes de las 7, puedes salir. | Puedes salir con la condición de que vengas a casa a las siete. | number_mismatch |
| 15 | 4.1 | What marvelous work he's doing, donating books to needy children! | ¡Qué maravilloso trabajo que está haciendo en donar libros a los niños necesitados! | ¡Qué trabajo tan maravilloso es hacer, no hacer libros que necesitan tener hijos! | other |
| 16 | 4.1 | Galician, Portuguese, and Spanish are often confused with one another. | A menudo se confunden el gallego, el portugués y el español. | Galician, portugués, y español a menudo son confundidos con uno al otro. | other |
| 17 | 4.2 | The electric bill keeps going up. | Cada vez sube más el precio de la electricidad. | La cuenta eléctrica sigue subiendo. | no_content_overlap |
| 18 | 4.4 | We make a teeny bit of progress, then we go back to square one. | Apenas logramos un avance, todo vuelve a cero. | Hacemos un poco de progreso, entonces volvimos a cuadrarnos a una. | no_content_overlap |
| 19 | 4.5 | It's coming out! My bladder's about to explode! | ¡Me estoy meando, la vejiga me va a explotar! | ¡Ya está fuera! ¡Mi vejiga está a punto de estallar! | other |
| 20 | 4.5 | My little sister has been suffering with a toothache since last night. | A mi hermana menor le duele un diente desde esta noche. | Mi hermana pequeña lleva con dolor de muelas desde anoche. | other |

**Failure-mode frequency across the whole test set**

| Category | Sentences | Rate |
|----------|-----------|------|
| other | 5357 | 94.7% |
| no_content_overlap | 215 | 3.8% |
| repetition | 33 | 0.6% |
| truncation | 24 | 0.4% |
| over_generation | 22 | 0.4% |
| number_mismatch | 12 | 0.2% |
| copied_source | 11 | 0.2% |


### Worst 20 translations — es-en

| # | sBLEU | Source | Reference | Model output | Failure modes |
|---|-------|--------|-----------|--------------|---------------|
| 1 | 0.0 | Aquí están diez dólares. | Here's $10.00. | Ten dollars are here. | number_mismatch, no_content_overlap |
| 2 | 0.0 | Te maldigo. | Curse you! | I swear. | no_content_overlap |
| 3 | 3.1 | Considera a las mujeres placeres de usar y tirar y no búsquedas con sentido. | He regards women as disposable pleasures rather than as meaningful pursuits. | Consider the women of pleasure to use and throw away and don't make sense. | other |
| 4 | 3.2 | Considera que las mujeres son placeres de usar y tirar más que búsquedas con sentido. | He regards women as disposable pleasures rather than as meaningful pursuits. | It seems that women are delights to use and throw more than dinner with meaning. | other |
| 5 | 3.7 | Considera a las mujeres placeres desechables y no propósitos con sentido. | He regards women as disposable pleasures rather than as meaningful pursuits. | Consider the women with disposal women and do not propose with meaning. | other |
| 6 | 4.0 | ¿Rusos nuevos, dices? ¡Maravilloso! ¡Es justo lo que necesitábamos! | New Russians, you say? Wonderful! That's just what we need! | You're new, you're saying? What we needed! | other |
| 7 | 4.1 | Un sólido argumento en favor del cristianismo es el siguiente: las ofensas contra la caridad es probablemente lo único que, en sus lechos de muerte, los hombres llegan a sentir y no a comprender como un crimen. | A strong argument for the religion of Christ is this - that offences against Charity are about the only ones which men on their death-beds can be made, not to understand, but to feel, as crime. | A solid argument in favor of the Christianity is the next: the offensive against unity is probably the only thing in his deaths, men came to feel and not understanding as a crime. | other |
| 8 | 4.1 | Estados Unidos se imagina que es la nación más libre del mundo. | The United States fancies itself the world's freest nation. | America is imagining that it is the free nation in the world. | other |
| 9 | 4.4 | Estados Unidos se imagina que es la nación más libre del mundo. | America fancies itself the world's freest nation. | America is imagining that it is the free nation in the world. | other |
| 10 | 4.5 | No apresure el paso; llegaremos a la hora. | Don't walk so fast. We'll get there on time. | Don't rush the step; we'll arrive at the hour. | other |
| 11 | 4.5 | Considera que las mujeres son placeres desechables más que objetivos significativos. | He regards women as disposable pleasures rather than as meaningful pursuits. | I consider women to be desirable more than mean goals. | other |
| 12 | 4.7 | Los baños de los aviones no tienen ventanillas por motivos de seguridad. | For security reasons, the bathrooms on airplanes don't have windows. | Airplanes have no windows for safety reasons. | other |
| 13 | 4.8 | Cada vez sube más el precio de la electricidad. | The electric bill keeps going up. | Every time it rises the price of electricity. | no_content_overlap |
| 14 | 4.8 | No hay falsedad en lo que él dice. | What he's saying is true. | There are no falsehood in what he says. | other |
| 15 | 4.8 | Como suele pasar, él no me trajo nada. | As usual he brought me nothing. | As is often the case, he didn't bring me anything. | over_generation |
| 16 | 4.9 | Ella puso en riesgo su vida para salvar a un niño de ahogarse. | She saved the drowning child at the risk of her own life. | She risked her life to save a child from drowning. | other |
| 17 | 4.9 | Tenga cuidado de no resbalar en las baldosas mojadas. | Mind you don't slip on the wet tiles. | Be careful not to slippery. | no_content_overlap |
| 18 | 4.9 | Por favor, ajuste su asiento como le acomode. | Please adjust the seat to fit you. | Please put your seat on as a hold of him. | other |
| 19 | 4.9 | Lo sentimos mucho por no poder ayudarlos. | We are sorry we can't help you. | We're very sorry for not being able to help them. | other |
| 20 | 4.9 | ¿Cuál fue el momento en el que te diste cuenta de que habías crecido? | When did you realize you've grown up? | What was the time you noticed that you had grown? | other |

**Failure-mode frequency across the whole test set**

| Category | Sentences | Rate |
|----------|-----------|------|
| other | 5449 | 96.3% |
| no_content_overlap | 129 | 2.3% |
| repetition | 33 | 0.6% |
| number_mismatch | 21 | 0.4% |
| truncation | 20 | 0.4% |
| over_generation | 15 | 0.3% |
| copied_source | 7 | 0.1% |
