"""
Health, Dietary, Exercise and Clinical Advisory Engine for Diabetes Risk Management
Provides customized nutritional blueprints, lifestyle regimens, workout plans, and medical precautions.
"""

def get_clinical_recommendations(risk_level, age=35, bmi=24.5, family_history="no", fingerprint_type="loop", ridge_count=18):
    risk_upper = str(risk_level).upper()

    # Base analysis
    bmi_category = "Normal Weight"
    if bmi < 18.5:
        bmi_category = "Underweight"
    elif bmi < 25.0:
        bmi_category = "Normal Weight"
    elif bmi < 30.0:
        bmi_category = "Overweight"
    else:
        bmi_category = "Obese"

    # Dermatoglyphic Clinical Insight
    fp_description = {
        "arch": "Arch patterns represent standard dermal ridge continuity. In dermatoglyphic epidemiology, simple arches combined with reduced ridge counts correlate with higher metabolic vulnerability when lifestyle stress is present.",
        "loop": "Loop patterns are the most common dermatoglyphic archetype. Radial or ulnar loops indicate balanced genetic phenotypic distribution with moderate metabolic susceptibility.",
        "whorl": "Whorl patterns represent high complexity triradii structures. Clinical studies note increased whorl frequencies in individuals with endocrine and hereditary glucose intolerance markers."
    }.get(fingerprint_type.lower(), "Standard dermatoglyphic biometric archetype.")

    # High Risk Plan
    if risk_upper in ['HIGH', 'ROSE', 'ELEVATED']:
        return {
            "status": "High Risk",
            "badge_color": "danger",
            "hex_color": "#ef4444",
            "summary": "Your biometric markers and clinical parameters indicate an elevated predisposition to diabetic or pre-diabetic states. Proactive intervention is strongly advised.",
            "dermatoglyphic_note": f"Pattern '{fingerprint_type.title()}' with Ridge Count {ridge_count}: {fp_description}",
            "bmi_evaluation": f"Your calculated BMI is {bmi:.1f} ({bmi_category}). Targeted weight optimization can reduce metabolic resistance by up to 58%.",
            "key_vitals": {
                "Target Fasting Blood Glucose": "< 100 mg/dL",
                "Target HbA1c": "< 5.7%",
                "Target Postprandial Glucose": "< 140 mg/dL",
                "Screening Frequency": "Every 3 Months"
            },
            "diet_plan": {
                "title": "Low-Glycemic Anti-Inflammatory Nutrition Protocol",
                "focus": "Complex low GI carbs, high soluble fiber, lean polyphenols, zero refined sugars.",
                "meals": [
                    {"time": "Breakfast (08:00 AM)", "items": "Steel-cut oatmeal with chia seeds, cinnamon, unsweetened almond milk, and a handful of walnuts or boiled egg whites."},
                    {"time": "Mid-Morning (11:00 AM)", "items": "Green tea / Matcha infusion with 5-6 soaked almonds or fresh cucumber slices with hummus."},
                    {"time": "Lunch (01:30 PM)", "items": "Quinoa / Brown rice (1 small cup) with steamed broccoli, spinach curry, grilled paneer/tofu/chicken breast, and cucumber-flaxseed raita."},
                    {"time": "Evening Snack (05:00 PM)", "items": "Roasted chickpeas (chana) or sprouted moong salad with lemon zest and pinch of rock salt."},
                    {"time": "Dinner (08:00 PM)", "items": "Clear vegetable broth / Lentil soup with sautéed zucchini, bell peppers, asparagus, and grilled fish or tofu."}
                ],
                "do_eat": ["Methi (fenugreek) seeds water", "Bitter gourd (Karela) extract", "Leafy cruciferous greens", "Berries in moderation", "Extra virgin olive oil", "Apple cider vinegar before meals"],
                "avoid": ["Sugar-sweetened beverages & sodas", "White bread, maida, and refined pastas", "Deep-fried snacks & processed meats", "High fructose fruit juices & syrup", "Trans-fat pastries and bakery confectioneries"]
            },
            "exercise_regimen": {
                "title": "Metabolic Conditioning & Insulin Sensitization Protocol",
                "weekly_goal": "Minimum 200–250 minutes of combined aerobic and resistance training.",
                "activities": [
                    {"type": "Zone 2 Cardio", "duration": "40 mins / 4 days a week", "desc": "Brisk power walking, stationary cycling, or swimming at conversational pace."},
                    {"type": "Resistance Training", "duration": "30 mins / 3 days a week", "desc": "Bodyweight squats, dumbbell lunges, resistance band push-ups to stimulate GLUT4 glucose transporter uptake in skeletal muscle."},
                    {"type": "Post-Meal Stroll", "duration": "10–15 mins after each meal", "desc": "Immediate blunting of postprandial blood glucose spikes through light ambulation."}
                ]
            },
            "herbal_supplements": [
                {"name": "Fenugreek (Trigonella foenum-graecum)", "usage": "1 tsp soaked overnight in water, consumed on empty stomach."},
                {"name": "Cinnamon (Ceylon)", "usage": "1/2 tsp pure Ceylon cinnamon in warm water or oatmeal to support insulin receptor sensitivity."},
                {"name": "Indian Gooseberry (Amla) + Turmeric", "usage": "Fresh amla juice with pinch of organic turmeric for beta-cell antioxidant protection."}
            ],
            "lifestyle_and_sleep": [
                "Strict circadian rhythm alignment: Sleep 7.5 to 8.5 hours in complete darkness to minimize cortisol-induced insulin resistance.",
                "Stress modulation: 15 minutes of Pranayama (Anulom Vilom / Box Breathing) daily.",
                "Hydration: 3.5 to 4 Liters of purified water per day.",
                "Immediate Clinical Action: Schedule a comprehensive HbA1c, Lipid Profile, and Fasting Insulin (HOMA-IR) test with an endocrinologist."
            ]
        }

    # Medium Risk Plan
    elif risk_upper == 'MEDIUM':
        return {
            "status": "Moderate Risk",
            "badge_color": "warning",
            "hex_color": "#f59e0b",
            "summary": "Moderate indicators detected. Early behavioral lifestyle optimization will easily reverse metabolic creep and prevent clinical progression.",
            "dermatoglyphic_note": f"Pattern '{fingerprint_type.title()}' with Ridge Count {ridge_count}: {fp_description}",
            "bmi_evaluation": f"Your calculated BMI is {bmi:.1f} ({bmi_category}). Maintaining a balanced body composition will protect pancreatic beta-cell reserve.",
            "key_vitals": {
                "Target Fasting Blood Glucose": "< 95 mg/dL",
                "Target HbA1c": "< 5.5%",
                "Target Postprandial Glucose": "< 130 mg/dL",
                "Screening Frequency": "Every 6 Months"
            },
            "diet_plan": {
                "title": "Balanced Whole-Food Glycemic Balance Diet",
                "focus": "High-fiber whole grains, prebiotic foods, clean proteins, and healthy fatty acid profile.",
                "meals": [
                    {"time": "Breakfast (08:30 AM)", "items": "Multigrain vegetable upma or whole-wheat avocado toast with 2 poached eggs / paneer scramble and fresh herbal tea."},
                    {"time": "Mid-Morning (11:30 AM)", "items": "One seasonal fruit (Green apple, Guava, or Berries) with 10 roasted pumpkin and sunflower seeds."},
                    {"time": "Lunch (01:30 PM)", "items": "Two multigrain rotis or brown rice bowl with mixed dal, sautéed green beans, fresh salad, and probiotic curd/yogurt."},
                    {"time": "Evening Snack (05:00 PM)", "items": "Makhana (Fox nuts) roasted in 1/2 tsp ghee with turmeric and pepper."},
                    {"time": "Dinner (08:00 PM)", "items": "Grilled vegetable quinoa stir fry or tofu/chicken salad with olive oil-lemon dressing."}
                ],
                "do_eat": ["Millets (Ragi, Jowar, Bajra, Foxtail)", "Raw salad before lunch & dinner", "Greek yogurt / kefir", "Nuts & seeds", "Flaxseed meal in chapati dough"],
                "avoid": ["Canned foods & artificial sweeteners", "Excessive potato and white rice portions", "Sweetened flavored yogurts", "Late-night snacking after 9 PM"]
            },
            "exercise_regimen": {
                "title": "Active Living & Cardiovascular Fitness",
                "weekly_goal": "150–180 minutes of moderate-to-vigorous physical activity.",
                "activities": [
                    {"type": "Aerobic Exercise", "duration": "30 mins / 5 days a week", "desc": "Jogging, brisk walking, badminton, or cycling."},
                    {"type": "Core & Strength", "duration": "25 mins / 2 days a week", "desc": "Planks, squats, lunges, and light kettlebell circuits."},
                    {"type": "Daily Step Target", "duration": "8,000 to 10,000 steps daily", "desc": "Consistent NEAT (Non-Exercise Activity Thermogenesis)."}
                ]
            },
            "herbal_supplements": [
                {"name": "Jamun Seed Powder", "usage": "1/2 tsp in lukewarm water before breakfast."},
                {"name": "Tulsi (Holy Basil) Infusion", "usage": "Fresh tulsi tea brewed with ginger and cardamom for metabolic boost."}
            ],
            "lifestyle_and_sleep": [
                "Maintain 7-8 hours restful sleep without blue light exposure 1 hr before bedtime.",
                "Keep a consistent meal schedule without prolonged irregular fasting.",
                "Practice active workplace breaks: Stand up every 45 minutes for 2 minutes of stretching."
            ]
        }

    # Low Risk Plan
    else:
        return {
            "status": "Low Risk",
            "badge_color": "success",
            "hex_color": "#10b981",
            "summary": "Excellent news! Your biometric fingerprint analysis and clinical variables show minimal diabetes susceptibility. Continue your proactive habits.",
            "dermatoglyphic_note": f"Pattern '{fingerprint_type.title()}' with Ridge Count {ridge_count}: {fp_description}",
            "bmi_evaluation": f"Your calculated BMI is {bmi:.1f} ({bmi_category}). Your current weight status aligns with healthy metabolic vitality.",
            "key_vitals": {
                "Target Fasting Blood Glucose": "< 90 mg/dL",
                "Target HbA1c": "< 5.4%",
                "Target Postprandial Glucose": "< 120 mg/dL",
                "Screening Frequency": "Annual Preventive Check"
            },
            "diet_plan": {
                "title": "Longevity & Preventive Wellness Diet",
                "focus": "Whole spectrum Mediterranean-style eating with abundant micronutrients and hydration.",
                "meals": [
                    {"time": "Breakfast", "items": "Fruit & nut smoothie bowl with spinach, banana, almond butter, and chia seeds OR vegetable multigrain paratha with fresh curd."},
                    {"time": "Lunch", "items": "Wholesome plate: 50% colorful veggies/salad, 25% lean protein/dal, 25% whole grains."},
                    {"time": "Evening", "items": "Green tea with roasted nuts or fruit salad."},
                    {"time": "Dinner", "items": "Light wholesome dinner with vegetable stir fry and soup or grilled protein."}
                ],
                "do_eat": ["Colorful rainbow vegetables", "Legumes & beans", "Cold-pressed oils", "Natural prebiotic foods", "Seasonal fresh fruits"],
                "avoid": ["Heavy ultra-processed fast food", "Excessive refined sugar snacking", "Chronic dehydration"]
            },
            "exercise_regimen": {
                "title": "Vitality, Mobility & Longevity Fitness",
                "weekly_goal": "150 minutes of enjoyable physical movement.",
                "activities": [
                    {"type": "Recreational Sports / Fitness", "duration": "30-45 mins / 4 days", "desc": "Swimming, tennis, hiking, dancing, or running."},
                    {"type": "Flexibility & Yoga", "duration": "20 mins / 3 days", "desc": "Surya Namaskar, stretching, and mobility routines."}
                ]
            },
            "herbal_supplements": [
                {"name": "Triphala Powder", "usage": "1/2 tsp before bedtime with warm water for gut health and detox."},
                {"name": "Herbal Tisane", "usage": "Chamomile, mint, or lemongrass tea."}
            ],
            "lifestyle_and_sleep": [
                "Sustain standard 7–8 hour sleep architecture.",
                "Yearly preventative health checkups.",
                "Mindfulness, regular laughter, outdoor nature exposure, and active hobbies."
            ]
        }
