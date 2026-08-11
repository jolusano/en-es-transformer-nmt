### Worst 20 translations — en-es

| # | sBLEU | Source | Reference | Model output | Failure modes |
|---|-------|--------|-----------|--------------|---------------|
| 1 | 0.0 | Tell me about it! | Ni me lo digas. | ¡Cuéntame! | truncation, no_content_overlap |
| 2 | 0.0 | Curse you! | Te maldigo. | ¡<unk>! | unknown_token, no_content_overlap |
| 3 | 0.0 | She will turn 15 tomorrow. | Ella cumplirá 15 años mañana. | Mañana <unk> 15. | unknown_token |
| 4 | 0.0 | Tell me about it. | ¡Dímelo a mí! | Cuéntame. | truncation, no_content_overlap |
| 5 | 2.1 | In footnotes, book titles and journal names are written in italics. | En las notas a pie de página, los títulos de los libros y los nombres de las revistas se escriben en cursiva. | En <unk>, el libro se escribe en <unk>. | repetition, truncation, unknown_token, no_content_overlap |
| 6 | 2.6 | Reduce to pulp by boiling. | Cocinar hasta convertir en pulpa. | reducir las <unk> <unk> <unk> <unk>. | repetition, over_generation, unknown_token, no_content_overlap |
| 7 | 2.8 | It's coming out! My bladder's about to explode! | ¡Me estoy meando, la vejiga me va a explotar! | ¡Se está saliendo! ¡Mi <unk> sobre que <unk>! | repetition, unknown_token, no_content_overlap |
| 8 | 3.4 | The team's best striker scored twenty-three goals last season. | El mejor delantero del equipo anotó veintitrés goles la temporada pasada. | El <unk> lo mejor marcó los goles de <unk> en la última temporada. | repetition, unknown_token |
| 9 | 3.4 | Ecosystems are degrading rapidly all around the Earth. | Los ecosistemas se están degradando en todo el mundo. | Los <unk> son <unk> rápidamente por toda la Tierra. | repetition, unknown_token, no_content_overlap |
| 10 | 3.6 | His meaning is quite plain. | Lo que él trata de decir es bien sencillo de comprender. | Su significado es bastante simple. | truncation, no_content_overlap |
| 11 | 3.7 | We make a teeny bit of progress, then we go back to square one. | Apenas logramos un avance, todo vuelve a cero. | <unk> un poco de progreso, entonces volvemos a la casilla uno. | unknown_token, no_content_overlap |
| 12 | 3.7 | The Geneva Convention is not recognized at Guantanamo. | La Convención de Ginebra no está reconocida en Guantanamo. | El tratado de <unk> no es reconocido en las <unk>. | repetition, unknown_token, no_content_overlap |
| 13 | 4.0 | You can go out on condition that you come home by seven. | Si vuelves antes de las 7, puedes salir. | Puedes salir con la condición de que vengas a casa a las siete. | number_mismatch |
| 14 | 4.0 | Wealthy older men often marry younger trophy wives. | Hombres ricos suelen casarse con jóvenes mujeres trofeo. | Los hombres ricos a menudo se casan con las mujeres <unk>. | unknown_token |
| 15 | 4.0 | Please adjust the seat to fit you. | Por favor, ajuste su asiento como le acomode. | <unk> el asiento para que te <unk>. | repetition, unknown_token |
| 16 | 4.0 | Do you have a lawn mower I could borrow? | ¿Tenés una cortadora de pasto para prestarme? | ¿Tienes un <unk> de césped que me puedas prestar? | unknown_token, no_content_overlap |
| 17 | 4.1 | Never throw away your notes - they may come in handy one day. | Nunca te deshagas de tus apuntes. Algún día te pueden resultar útiles. | Nunca <unk> tus notas, pueden pasar a mano. | unknown_token |
| 18 | 4.1 | And passing on from thence to a mountain that was on the east side of Bethel, he there pitched his tent, having Bethel on the west and Hai on the east: he built there also an altar to the Lord, and called upon his name. | De allí pasó a la montaña, al oriente de Betel, y desplegó su tienda, entre Betel al occidente y Ay al oriente. Allí edificó un altar a Yahvé e invocó su nombre. | Y <unk> de <unk> a una montaña que estaba en el este lado de la <unk>, él allí <unk> su carpa, habiendo <unk> en el oeste y en el Este: él también construyó un altar para Dios y llamó a su nombre. | repetition, unknown_token |
| 19 | 4.1 | You can meet me somewhere sometime or something. | Podemos quedar en algún sitio en algún momento, o algo. | Puedes encontrarte conmigo alguna vez en alguna parte. | no_content_overlap |
| 20 | 4.1 | Artsakh has never belonged to Azerbaijan. | Artsaj nunca perteneció a Azerbaiyán. | <unk> nunca ha <unk> a <unk>. | repetition, over_generation, unknown_token |

**Failure-mode frequency across the whole test set**

| Category | Sentences | Rate |
|----------|-----------|------|
| other | 4367 | 77.2% |
| unknown_token | 1101 | 19.5% |
| no_content_overlap | 231 | 4.1% |
| repetition | 192 | 3.4% |
| over_generation | 87 | 1.5% |
| truncation | 22 | 0.4% |
| number_mismatch | 15 | 0.3% |
| copied_source | 10 | 0.2% |


### Worst 20 translations — es-en

| # | sBLEU | Source | Reference | Model output | Failure modes |
|---|-------|--------|-----------|--------------|---------------|
| 1 | 0.0 | Aquí están diez dólares. | Here's $10.00. | There are ten dollars here. | over_generation, number_mismatch, no_content_overlap |
| 2 | 3.1 | Marque 411 para asistencia de directorio. | Dial 411 for directory assistance. | <unk> <unk> <unk> <unk>. | repetition, over_generation, unknown_token, number_mismatch, no_content_overlap |
| 3 | 3.2 | Estados Unidos se imagina que es la nación más libre del mundo. | America fancies itself the world's freest nation. | The United States is imagining that it is the most free nation in the world. | over_generation |
| 4 | 3.4 | Aprovechan la sociedad del miedo para sus propios fines aviesos. | They take advantage of society for their own malicious goals. | He <unk> the society of fear for his own fines of <unk>. | repetition, unknown_token |
| 5 | 3.6 | Apenas logramos un avance, todo vuelve a cero. | We make a teeny bit of progress, then we go back to square one. | We barely achieve a advance, everything comes to zero. | no_content_overlap |
| 6 | 3.7 | Considera que las mujeres son placeres de usar y tirar más que búsquedas con sentido. | He regards women as disposable pleasures rather than as meaningful pursuits. | Consider women are pleasures to use and throw more than to make them sense. | other |
| 7 | 3.7 | Considera a las mujeres placeres de usar y tirar y no búsquedas con sentido. | He regards women as disposable pleasures rather than as meaningful pursuits. | Consider women pleasures to use and throw away and don't make it sense. | other |
| 8 | 3.7 | ¿Podrás hacerlo para la fecha tope? | Can you make the deadline? | Will you be able to do it for the <unk> date? | over_generation, unknown_token, no_content_overlap |
| 9 | 4.0 | Considera que las mujeres son placeres desechables más que objetivos significativos. | He regards women as disposable pleasures rather than as meaningful pursuits. | Consider women are <unk> pleasures more than <unk> goals. | repetition, unknown_token |
| 10 | 4.0 | Por favor incluya un sobre estampillado a su domicilio con su envio. | Please include a self-addressed, stamped envelope with your submission. | Please put an envelope over to his address with his <unk>. | unknown_token |
| 11 | 4.0 | Están debatiendo acerca de los aumentos de sueldo. | They're debating about raises. | They are <unk> about the salary <unk>. | repetition, over_generation, unknown_token |
| 12 | 4.1 | Tuve mucha, pero que mucha suerte. | I was very, very lucky. | I had a lot, but I had a lot of luck. | repetition, over_generation, no_content_overlap |
| 13 | 4.1 | Tenga cuidado de no resbalar en las baldosas mojadas. | Mind you don't slip on the wet tiles. | Be careful not to <unk> on wet <unk>. | repetition, unknown_token |
| 14 | 4.1 | En la tienda sólo admiten pagos en metálico. | The shop only admits payments in cash. | In the store, there are only <unk> in metal. | unknown_token |
| 15 | 4.2 | ¿Es cosa tuya cuál sea el pasatiempo de una persona? | Is it any of your business what someone's hobby is? | What's your sister's hobby? | truncation |
| 16 | 4.3 | Un sólido argumento en favor del cristianismo es el siguiente: las ofensas contra la caridad es probablemente lo único que, en sus lechos de muerte, los hombres llegan a sentir y no a comprender como un crimen. | A strong argument for the religion of Christ is this - that offences against Charity are about the only ones which men on their death-beds can be made, not to understand, but to feel, as crime. | A solid argument in favor of Christianity is the following: the trespasses against charity is probably the only thing that, in their death, men come to feel and not understand as a crime. | other |
| 17 | 4.3 | La inteligencia artificial amenaza con extinguir los puestos de trabajo tanto de personas con estudios bajos como de personas con estudios superiores. | Artificial intelligence threatens to extinguish the jobs of both low-educated and highly-educated people. | Artificial intelligence is threatening to put away on working as much as people with short studies as people with higher studies. | repetition |
| 18 | 4.3 | Esto viola alrededor de un millón de códigos penales y todos los libros sagrados existentes. | This violates about a million penal codes and every holy book there is. | This <unk> around a million <unk> and all of the books <unk> <unk>. | repetition, unknown_token |
| 19 | 4.4 | Si vuelves antes de las 7, puedes salir. | You can go out on condition that you come home by seven. | If you go back before 7, you can leave. | other |
| 20 | 4.5 | Lo que él trata de decir es bien sencillo de comprender. | His meaning is quite plain. | What he tries to say is well simple to understand. | over_generation, no_content_overlap |

**Failure-mode frequency across the whole test set**

| Category | Sentences | Rate |
|----------|-----------|------|
| other | 4781 | 84.5% |
| unknown_token | 687 | 12.1% |
| no_content_overlap | 176 | 3.1% |
| repetition | 138 | 2.4% |
| over_generation | 40 | 0.7% |
| number_mismatch | 23 | 0.4% |
| truncation | 10 | 0.2% |
| copied_source | 4 | 0.1% |
