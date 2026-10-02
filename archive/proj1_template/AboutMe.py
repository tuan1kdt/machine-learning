import streamlit as st
import streamlit_shadcn_ui as ui

def AboutMe():
    # If set_page_config is already in Main.py, remove this:
    # st.set_page_config(page_title="About • An", page_icon="🤖", layout="wide")

    # --- Shadcn-ish cards + simple animation using CSS (no ui.card) ---
    st.markdown(
        """
        <style>
          .fade-in-up { animation: fadeInUp .55s ease-out both; }
          @keyframes fadeInUp {
            from { opacity: 0; transform: translate3d(0, 10px, 0); }
            to   { opacity: 1; transform: translate3d(0, 0, 0); }
          }

          .card {
            border: 1px solid rgba(0,0,0,.10);
            border-radius: 16px;
            padding: 18px 18px;
            background: rgba(255,255,255,.70);
          }
          /* Dark mode support */
          @media (prefers-color-scheme: dark) {
            .card { background: rgba(20,20,20,.35); border-color: rgba(255,255,255,.10); }
          }

          .muted { opacity: .75; }
          .pill {
            display:inline-block;
            padding:.28rem .65rem;
            border-radius: 999px;
            border: 1px solid rgba(0,0,0,.12);
            margin: .18rem .22rem 0 0;
            font-size: .86rem;
            white-space: nowrap;
          }
          @media (prefers-color-scheme: dark) {
            .pill { border-color: rgba(255,255,255,.14); }
          }

          .section-title { font-weight: 650; font-size: 1.1rem; margin: 0 0 .35rem 0; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # --- HERO ---
    st.markdown(
        """
        <div class="card fade-in-up">
          <div style="display:flex;align-items:center;gap:.6rem;flex-wrap:wrap;">
            <div style="font-size:1.35rem;">✨</div>
            <div style="font-size:1.75rem;font-weight:700;">An</div>
            <div class="muted">• 18 years old</div>
          </div>
          <div class="muted" style="margin-top:.4rem; font-size:1.02rem; line-height:1.5;">
            I’m an enthusiastic learner in <b>Machine Learning</b> and <b>Artificial Intelligence</b>.
            I love turning ideas into practical, data-driven products — from modeling to clean UI.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    # --- METRICS (shadcn) ---
    c1, c2, c3, c4 = st.columns(4, gap="large")
    with c1:
        ui.metric_card(title="Focus", content="ML / AI", description="Learning • Building • Improving", key="m1")
    with c2:
        ui.metric_card(title="Strength", content="Python", description="Fast iteration & clean code", key="m2")
    with c3:
        ui.metric_card(title="Mindset", content="Curious", description="Always exploring", key="m3")
    with c4:
        ui.metric_card(title="Portfolio", content="Projects", description="Demos & experiments", key="m4")

    st.write("")

    left, right = st.columns([1.6, 1], gap="large")

    with left:
        # About
        st.markdown(
            """
            <div class="card fade-in-up">
              <div class="section-title">👋 About</div>
              <div class="muted" style="line-height:1.55;">
                I’m <b>An</b>, 18. I’m passionate about <b>Machine Learning</b> and <b>AI</b>,
                especially building real applications people can use.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")

        # Skills (pills)
        st.markdown(
            """
            <div class="card fade-in-up">
              <div class="section-title">🧠 Skills</div>
              <div>
                <span class="pill">Machine Learning</span>
                <span class="pill">Deep Learning</span>
                <span class="pill">Computer Vision</span>
                <span class="pill">NLP</span>
                <span class="pill">Data Analysis</span>
                <span class="pill">Model Evaluation</span>
                <span class="pill">MLOps (learning)</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")

        # Projects (template)
        st.markdown(
            """
            <div class="card fade-in-up">
              <div class="section-title">🚀 Featured Projects</div>
              <div class="muted">Replace these with your real projects.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        p1, p2 = st.columns(2, gap="medium")
        with p1:
            st.markdown(
                """
                <div class="card fade-in-up">
                  <div style="font-weight:650;">📌 Project 1</div>
                  <div class="muted" style="margin-top:.35rem;">
                    One-line description: what it does + model/approach.
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            # Use Streamlit button/links (safe)
            st.link_button("View Demo", "#")

        with p2:
            st.markdown(
                """
                <div class="card fade-in-up">
                  <div style="font-weight:650;">📌 Project 2</div>
                  <div class="muted" style="margin-top:.35rem;">
                    One-line description: dataset + metrics/outcome.
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.link_button("View Code", "#")

    with right:
        # Links
        st.markdown(
            """
            <div class="card fade-in-up">
              <div class="section-title">🔗 Links</div>
              <div class="muted">Update these with your real URLs.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.link_button("💻 GitHub", "#")
        st.link_button("🔗 LinkedIn", "#")
        st.link_button("🌐 Portfolio", "#")
        st.link_button("📄 Resume / CV", "#")

        st.write("")

        # CTA
        st.markdown(
            """
            <div class="card fade-in-up">
              <div class="section-title">🤝 Collaboration</div>
              <div class="muted">Open to ML/AI collaborations and hackathons.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.button("Say hi 👋", key="say_hi")
