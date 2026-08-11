### Worst 20 translations — en-es

| # | sBLEU | Source | Reference | Model output | Failure modes |
|---|-------|--------|-----------|--------------|---------------|
| 1 | 0.0 | Sit down! | ¡Sentaos! | Siéntate. | no_content_overlap |
| 2 | 0.0 | Drink! | ¡Bebe! | Bebe. | no_content_overlap |
| 3 | 0.0 | Curse you! | Te maldigo. | ¡Te agradezco! | no_content_overlap |
| 4 | 0.0 | Don't screw around with me. | ¡No jodas conmigo! | No me des la vuelta. | no_content_overlap |
| 5 | 0.0 | Sit down! | ¡Sentate! | Siéntate. | no_content_overlap |
| 6 | 0.0 | Tell me about it! | ¡Dímelo a mí! | Cuéntame al respecto. | no_content_overlap |
| 7 | 0.0 | Tell me about it. | ¡Dímelo a mí! | Cuéntame sobre ello. | no_content_overlap |
| 8 | 0.0 | Goodbye for ever! | ¡Adiós para siempre! | Adiós nunca. | no_content_overlap |
| 9 | 0.0 | Sit down! | ¡Siéntate! | Siéntate. | no_content_overlap |
| 10 | 0.0 | Speak slowly! | ¡Hable lentamente! | Habla despacio. | no_content_overlap |
| 11 | 3.0 | A strong argument for the religion of Christ is this - that offences against Charity are about the only ones which men on their death-beds can be made, not to understand, but to feel, as crime. | Un sólido argumento en favor del cristianismo es el siguiente: las ofensas contra la caridad es probablemente lo único que, en sus lechos de muerte, los hombres llegan a sentir y no a comprender como un crimen. | Un fuerte argumento de la religión de Cristo es esta - que las de las cosas en contra de la Carualidad se pueden hacer, sino que para sentir, como el crimen. | repetition |
| 12 | 3.6 | His meaning is quite plain. | Lo que él trata de decir es bien sencillo de comprender. | Su significado es bastante claro. | truncation, no_content_overlap |
| 13 | 4.0 | You can go out on condition that you come home by seven. | Si vuelves antes de las 7, puedes salir. | Puedes salir con la condición de que vengas a casa a las siete. | number_mismatch |
| 14 | 4.1 | My little sister has been suffering with a toothache since last night. | A mi hermana menor le duele un diente desde esta noche. | Mi hermana pequeña ha estado sufriendo con dolor de muelas desde anoche. | other |
| 15 | 4.2 | The electric bill keeps going up. | Cada vez sube más el precio de la electricidad. | La cuenta eléctrica sigue subiendo. | no_content_overlap |
| 16 | 4.3 | Never throw away your notes - they may come in handy one day. | Nunca te deshagas de tus apuntes. Algún día te pueden resultar útiles. | Nunca tires tus notas, pueden llegar a la mano un día. | other |
| 17 | 4.3 | That would be putting the cart before the horse. | Sería como poner el carro delante de los bueyes. | Eso pondría la carreta antes del caballo. | no_content_overlap |
| 18 | 4.4 | I nodded to show that I agreed. | Asentí con la cabeza para mostrar que estaba de acuerdo. | Asistí a demostrar que he acordado. | no_content_overlap |
| 19 | 4.5 | How do you tighten this screw? Clockwise or counterclockwise? | ¿Cómo se aprieta este tornillo? ¿En el sentido de las agujas del reloj o en el sentido contrario a las agujas del reloj? | ¿Cómo aprietas este tornillo? ¿Hinucible o sentido sentido? | truncation |
| 20 | 4.5 | Wealthy older men often marry younger trophy wives. | Hombres ricos suelen casarse con jóvenes mujeres trofeo. | A menudo los hombres mayores frecuentemente se casan esposas jóvenes. | other |

**Failure-mode frequency across the whole test set**

| Category | Sentences | Rate |
|----------|-----------|------|
| other | 5313 | 93.9% |
| no_content_overlap | 260 | 4.6% |
| truncation | 41 | 0.7% |
| repetition | 29 | 0.5% |
| over_generation | 16 | 0.3% |
| number_mismatch | 10 | 0.2% |
| copied_source | 7 | 0.1% |


### Worst 20 translations — es-en

| # | sBLEU | Source | Reference | Model output | Failure modes |
|---|-------|--------|-----------|--------------|---------------|
| 1 | 0.0 | Aquí están diez dólares. | Here's $10.00. | Here are ten dollars. | number_mismatch, no_content_overlap |
| 2 | 2.8 | Considera que las mujeres son placeres de usar y tirar más que búsquedas con sentido. | He regards women as disposable pleasures rather than as meaningful pursuits. | Consider that women are glad to use and throw more than you would be made of meaning. | other |
| 3 | 2.8 | Considera a las mujeres placeres de usar y tirar y no búsquedas con sentido. | He regards women as disposable pleasures rather than as meaningful pursuits. | Consider women's glad to use and throw away and don't get into meaning. | no_content_overlap |
| 4 | 3.3 | Apenas logramos un avance, todo vuelve a cero. | We make a teeny bit of progress, then we go back to square one. | We can hardly make an avance, everything returns. | other |
| 5 | 3.4 | Considera a las mujeres placeres desechables y no propósitos con sentido. | He regards women as disposable pleasures rather than as meaningful pursuits. | Consider women's pleasureable and don't propense in meaning. | no_content_overlap |
| 6 | 4.0 | Hace muchísimos años que él no celebra la Navidad. | He hasn't celebrated Christmas in so many years. | It's been a lot of years that he does not celebrate. | other |
| 7 | 4.0 | Placer y alegría tan pronto ida como venida. | Happiness comes and goes. | Famure and joy as soon as they come into the coming. | over_generation, no_content_overlap |
| 8 | 4.2 | Cada vez sube más el precio de la electricidad. | The electric bill keeps going up. | More and the price of electricity is every time. | no_content_overlap |
| 9 | 4.3 | El Atlético de Madrid sorprendió a todos con su victoria frente al Barcelona, ya que siempre ha sido uno de los tapados de la liga española. | Atletico Madrid surprised everyone with their win against Barcelona as they've always been the underdogs in the Spanish league. | Madrid's Atletic surprised all of his victory in front of Barcelona, since he has always been one of the tapes of Spanish lower. | other |
| 10 | 4.5 | Considera que las mujeres son placeres desechables más que objetivos significativos. | He regards women as disposable pleasures rather than as meaningful pursuits. | Consider that women are pleasure more than significations. | other |
| 11 | 4.6 | ¡Démelo o le embrisco al perro! | Hand it over or I'm siccing the dog on you! | Give it to my dog! | truncation |
| 12 | 4.7 | La sonda espacial rusa Luna 3 vio el lado oculto de la luna por primera vez en 1959. | The Russian space probe Luna 3 saw the far side of the moon for the first time in 1959. | The Russian Russian probe in 1959, in 1959. | truncation, number_mismatch |
| 13 | 4.8 | Me parece que el tiempo se está despejando. | I think it's clearing up. | It seems to me that time is clear. | no_content_overlap |
| 14 | 4.8 | La cárcel sirve para disuadir de cometer delitos a los delincuentes. | Prison serves to deter criminals from committing crimes. | prison is hard for deterting the criminal's offends. | other |
| 15 | 4.8 | Salgo de la casa a las ocho y cuarto, y llego al colegio a las nueve menos cuarto. | I leave the house at 8.15 and arrive at school at 8.45. | I go out from home at eight past eight, and I get to school at nine least room. | number_mismatch |
| 16 | 4.9 | Ella puso en riesgo su vida para salvar a un niño de ahogarse. | She saved the drowning child at the risk of her own life. | She risked her life to save a child from drowning. | other |
| 17 | 5.0 | Lo que él trata de decir es bien sencillo de comprender. | His meaning is quite plain. | What he tries to say is simple to understand. | no_content_overlap |
| 18 | 5.0 | Después de conferenciar varias horas hicieron públicos los acuerdos. | After conferring for several hours, they made the agreements public. | After reading many hours made public publics. | other |
| 19 | 5.0 | El odio no es algo que surge de la nada, habitualmente nace de la envidia o del miedo. | Hatred doesn't just appear out of thin air; it usually starts from envy or fear. | Hate isn't something that's beginning of nothing, mostly born and fear. | other |
| 20 | 5.1 | ¡Me estoy meando, la vejiga me va a explotar! | It's coming out! My bladder's about to explode! | I'm peeing, my bladder will explode me! | other |

**Failure-mode frequency across the whole test set**

| Category | Sentences | Rate |
|----------|-----------|------|
| other | 5349 | 94.6% |
| no_content_overlap | 221 | 3.9% |
| repetition | 35 | 0.6% |
| truncation | 33 | 0.6% |
| number_mismatch | 19 | 0.3% |
| over_generation | 9 | 0.2% |
| copied_source | 5 | 0.1% |
